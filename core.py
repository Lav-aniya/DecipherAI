import os
import json
import numpy as np

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.embeddings import SentenceTransformerEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def generate_question_and_benchmark(skill, topic, difficulty, candidate_profile):
    """
    Dynamically generates a question, benchmark answer, and rubric.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    print(f"\n--- Generating a {difficulty} question for {skill} on {topic}... ---")

    system_prompt = """
    You are an expert technical interviewer. Your task is to generate a single, high-quality interview question.
    You must return the output in a clean JSON format with three keys: "question", "benchmark_answer", and "rubric".
    The "rubric" should be a list of dictionaries, where each dictionary has a "concept" and "points" key.
    """
    context_str = json.dumps(candidate_profile, indent=2)
    user_prompt_template = f"""
    Please generate a question based on the following criteria:
    - Skill: {skill}
    - Topic: {topic}
    - Difficulty: {difficulty}
    - Candidate Context: {candidate_profile}

    Ensure the question is relevant to the candidate's background and the specified difficulty.
    The benchmark answer should be what you'd expect from an ideal candidate.
    """

    model = ChatGroq(model_name = "llama-3.3-70b-versatile",
                     groq_api_key=GROQ_API_KEY)
    parser = JsonOutputParser

    prompt = ChatPromptTemplate.format_messages([
        ("system", system_prompt),
        ("human", user_prompt_template)
    ])

    json_model = model.bind(response_format={"type": "json_object"})
    chain = prompt | json_model | parser

    try:
        response = chain.invoke({
            "skill": skill,
            "topic": topic,
            "difficulty": difficulty,
            "context": context_str
        })
        return response
    except Exception as e:
        print(f"Error generating question with LangChain/Groq: {e}")
        return None
    
def get_embedding(text):
    """
    Gets the embedding for a piece of text using a local Sentence Transformer model.
    This runs on your machine and doesn't require an API call.
    """
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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

    for item in benchmark_data['rubric']:
        concept_text = item['concept']
        concept_points = item.get('points', 0)

        concept_embedding = get_embedding(concept_text)

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