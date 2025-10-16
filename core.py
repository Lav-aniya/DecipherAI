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

    # 1. Instantiate the Pydantic parser with your desired data structure
    parser = PydanticOutputParser(pydantic_object=InterviewQuestion)

    system_prompt = """
    You are an expert technical interviewer. Your task is to generate a single, high-quality interview question.
    You must return the output in a clean JSON format with three keys: "question", "benchmark_answer", and "rubric".
    The "rubric" should be a list of dictionaries, where each dictionary has a "concept" and "points" key.
    {format_instructions}
    """
    #context_str = json.dumps(candidate_profile, indent=2)
    user_prompt_template = """
    Please generate a question based on the following criteria:
    - Skill: {skill}
    - Topic: {topic}
    - Difficulty: {difficulty}
    - Candidate Context: {candidate_profile}

    Ensure the question is relevant to the candidate's background and the specified difficulty.
    The benchmark answer should be what you'd expect from an ideal candidate.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt_template)
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
    
def get_embedding(text):
    """
    Gets the embedding for a piece of text using a local Sentence Transformer model.
    This runs on your machine and doesn't require an API call.
    """
    return embedding_model.embed_query(text)

def evaluate_answer(candidate_answer, benchmark_data):
    """
    Evaluates the candidate's answer against the benchmark 
    """
    print("\n--- Performing granular evaluation... ---")

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