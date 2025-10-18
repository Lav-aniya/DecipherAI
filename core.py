import os
import json
import numpy as np

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Load the embedding model once when the script starts.
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initialize the chat model once.
chat_model = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=GROQ_API_KEY
)


# Define the structure for a single item in the rubric
class RubricItem(BaseModel):
    concept: str = Field(description="The specific concept or skill being evaluated.")
    points: int = Field(description="The number of points this concept is worth.")

# Define the main structure for the entire JSON output
class InterviewQuestion(BaseModel):
    question: str = Field(description="The text of the interview question.")
    benchmark_answer: str = Field(description="A detailed, ideal answer to the question.")
    rubric: List[RubricItem] = Field(description="A list of concepts for grading the answer.")

def generate_question_and_benchmark(skill, topic, difficulty, candidate_profile):

    """
    Dynamically generates a question, benchmark answer, and rubric.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    print(f"\n--- Generating a {difficulty} question for {skill} on {topic}... ---")

    # Instantiate the Pydantic parser with your desired data structure
    parser = PydanticOutputParser(pydantic_object=InterviewQuestion)

    system_prompt = """
    You are an expert technical interviewer. Your task is to generate a single, high-quality interview question.
    You must return the output in a clean JSON format with three keys: "question", "benchmark_answer", and "rubric".
    The "rubric" should be a list of dictionaries, where each dictionary has a "concept" and "points" key.
    {format_instructions}
    """
    #context_str = json.dumps(candidate_profile, indent=2)
    # user_prompt = """
    # Please generate a question based on the following criteria:
    # - Skill: {skill}
    # - Topic: {topic}
    # - Difficulty: {difficulty}
    # - Candidate Context: {candidate_profile}

    # Ensure the question is relevant to the candidate's background and the specified difficulty.
    # The benchmark answer should be what you'd expect from an ideal candidate.
    # """

    if topic.startswith("PROBE_SECTION:"):
        try:
            section_to_probe = topic.split(":")[1]
            section_data = candidate_profile.get(section_to_probe)

            if not section_data:
                print(f"--> Section '{section_to_probe}' not found or empty. Asking a general question instead.")
                user_prompt = "Please generate a general question for the skill {skill} with difficulty {difficulty}."
            else:
                print(f"--> Using specialized prompt to probe the '{section_to_probe}' section.")
                user_prompt = f"""
                Please generate a question that probes the candidate's experience based on their resume.
                - Section to Probe: {section_to_probe}
                - Skill to Relate to: {skill}
                - Candidate's Data for this Section: {json.dumps(section_data, indent=2)}
                
                Instructions: Analyze the data and formulate one specific, open-ended question about an entry, relating it back to the specified skill ({skill}).
                """
        except IndexError:
            print(f"--> Invalid probe format '{topic}'. Asking a general question.")
            user_prompt = "Please generate a general question for the skill {skill}."
    else:
        # Your original user_prompt now lives inside the 'else' block
        user_prompt = """
        Please generate a question based on the following criteria:
        - Topic: {topic}
        - Skill: {skill}
        - Difficulty: {difficulty}
        - Candidate Context: {candidate_profile}

        Ensure the question is relevant to the candidate's background and the specified difficulty.
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ]).partial(format_instructions=parser.get_format_instructions())

    #parser = JsonOutputParser
    #json_model = chat_model.bind(response_format={"type": "json_object"})
    chain = prompt | chat_model | parser

    try:
        response = chain.invoke({
            "skill": skill,
            "topic": topic,
            "difficulty": difficulty,
            "candidate_profile": json.dumps(candidate_profile, indent=2)
        })
        #return response
        return response.dict()
    except Exception as e:
        print(f"Error generating question with LangChain/Groq: {e}")
        return None


# --- Pydantic Data Structures for Evaluation ---
class EvaluationItem(BaseModel):
    concept: str = Field(description="The concept from the original rubric.")
    covered: bool = Field(description="Whether the candidate's answer covered this concept.")
    justification: str = Field(description="A brief justification for the decision.")
    earned_points: int = Field(description="Points awarded for this specific concept.")

class InterviewEvaluation(BaseModel):
    evaluation_summary: str = Field(description="A one-sentence overall summary of the candidate's answer.")
    evaluation_breakdown: List[EvaluationItem] = Field(description="A detailed breakdown of the evaluation against the rubric.")

# NOT USING THIS 
def get_embedding(text):
    """
    Gets the embedding for a piece of text using a local Sentence Transformer model.
    This runs on your machine and doesn't require an API call.
    """
    return embedding_model.embed_query(text)

# UPDATED EVALUATION
# Using LLM as a Judge now, no more similarity scores
def evaluate_answer(candidate_answer, benchmark_data):
    """
    Evaluates the candidate's answer against the benchmark 
    """
    print("\n--- Performing evaluation... ---")

    parser = PydanticOutputParser(pydantic_object=InterviewEvaluation)

    system_prompt = """
    You are a fair, strict, and expert technical interviewer. Your task is to evaluate a candidate's answer.
    Analyze the provided information and determine if the candidate's answer covers the concepts in the rubric.
    Provide a brief justification for your decision on each concept.
    You must format your output as a JSON object that adheres to the provided schema.
    {format_instructions}
    """

    user_prompt = """
    Please evaluate the following interview response:

    **Original Question:**
    {question}

    **Ideal Benchmark Answer:**
    {benchmark_answer}

    **Scoring Rubric:**
    {rubric}

    ---
    **CANDIDATE'S ANSWER:**
    {candidate_answer}
    ---

    Evaluate the candidate's answer against each concept in the rubric.
    For each concept, set 'covered' to true if the candidate addressed it adequately, and false otherwise.
    Award points accordingly. Provide a concise justification for each decision.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | chat_model | parser

    try:
        # pass all context to the LLM for evaluation
        response = chain.invoke({
            "question": benchmark_data['question'],
            "benchmark_answer": benchmark_data['benchmark_answer'],
            "rubric": json.dumps(benchmark_data['rubric'], indent=2),
            "candidate_answer": candidate_answer
        })

        # pRoceess the structured response to create the final feedback
        earned_points = sum(item.earned_points for item in response.evaluation_breakdown)
        total_points = sum(item.get('points', 0) for item in benchmark_data['rubric'])

        feedback_parts = [response.evaluation_summary]

        strengths = [item for item in response.evaluation_breakdown if item.covered]
        improvements = [item for item in response.evaluation_breakdown if not item.covered]

        if strengths:
            feedback_parts.append("\n" + "**What you covered well:**")
            for item in strengths:
                feedback_parts.append(f"  - **{item.concept}:** {item.justification}")
        
        if improvements:
            feedback_parts.append("\n" + "**Areas for improvement:**")
            for item in improvements:
                feedback_parts.append(f"  - **{item.concept}:** {item.justification}")
                
        return {
            "earned_points": earned_points,
            "total_points": total_points,
            "feedback": "\n".join(feedback_parts)
        }
    
    except Exception as e:
        print(f"Error evaluating answer with AI Adjudicator: {e}")
        return {
            "earned_points": 0,
            "total_points": sum(item.get('points', 0) for item in benchmark_data['rubric']),
            "feedback": "An error occurred during evaluation."
        }

    '''

    # Evaluation usin cosine_similarity

    SIMILARITY_THRESHOLD = 0.5

    candidate_embedding = get_embedding(candidate_answer)

    earned_points = 0
    total_points = sum(item.get('points', 0) for item in benchmark_data['rubric'])
    
    strengths = []
    improvements = []

    # Efficiently embed all rubric concepts at once.
    rubric_concepts = [item['concept'] for item in benchmark_data['rubric']]
    rubric_embeddings = embedding_model.embed_documents(rubric_concepts)

    for i,item in enumerate(benchmark_data['rubric']):
        concept_text = item['concept']
        concept_points = item.get('points', 0)

        concept_embedding = rubric_embeddings[i]

        similarity_score = cosine_similarity(
            np.array(candidate_embedding).reshape(1, -1),
            np.array(concept_embedding).reshape(1, -1)
        )[0][0]

        if similarity_score >= SIMILARITY_THRESHOLD:
            earned_points += concept_points
            strengths.append(concept_text)
        else:
            improvements.append(concept_text)

    feedback_parts = []
    if not strengths:
        feedback_parts.append("the answer did not seem to cover the key concepts expected for this question")
    else:
        feedback_parts.append(" **What you covered well:**")
        for strength in strengths:
            feedback_parts.append(f"  - {strength}")
    
    if improvements:
        feedback_parts.append("\n" + "**Areas for improvement:**")
        for improvement in improvements:
            feedback_parts.append(f" - Your answer could be stronger by including or elaborating on: {improvement}")
    

    return {
        "earned_points": round(earned_points),
        "total_points": total_points,
        "feedback": "\n".join(feedback_parts)
    }

    '''