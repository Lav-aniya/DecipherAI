from pypdf import PdfReader
import re
import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

reader = PdfReader("Aditya_Lavaniya_Resume.pdf")
page = reader.pages[0]
text = page.extract_text()
#print(text)

load_dotenv(dotenv_path="secrets.env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def parse_resume(resume_text):
    if not GROQ_API_KEY :
        raise ValueError("API key not found")
    
    print("--- Parsing resume text with AI.. ---")

    system_prompt = """
    You are an expert HR recruitment assistant specializing in tech roles.
    Your task is to parse the provided resume text and extract key information in a structured JSON format.
    The JSON object must contain the following keys: "name", "skills", "total_experience", and "projects".
    If there is any other useful section like blogs, awards and achievements, scholarships add those as well.
    "projects" should be a list of strings.
    "skills" should be a list of relevant technical skills.
    Include internships in Experience as well.
    Calculate the total experience in months if it exceeds 12 months use years based on the dates provided.
    
    """
    model = ChatGroq(model_name = "llama-3.3-70b-versatile",
                     groq_api_key=GROQ_API_KEY)
    # this will automatically parse the model's JSON string output into a python dicttionary
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{resume_text}")
    ])

    json_model = model.bind(response_format={"type": "json_object"})

    chain = prompt | json_model | parser
    
    try:
        candidate_profile = chain.invoke({"resume_text": resume_text})
        return candidate_profile
    except Exception as e:
        print(f"Error parsing resume with langchain: {e}")
        return None
    
if __name__ == "__main__":
    profile = parse_resume(text)
    if profile:
        print("\n -- Successfulyy Parsed Candidate Profile --")
        print(json.dumps(profile, indent=2))
