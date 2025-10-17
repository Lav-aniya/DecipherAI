import random
from pypdf import PdfReader
import json

from prof import parse_resume
from Interview_config import setup_interview_configuration, JOB_ROLES_CONFIG
from core import generate_question_and_benchmark, evaluate_answer

def select_role():
    """Allows the user to interactively select a role from the configuration"""
    print("select a role for interview")
    roles = list(JOB_ROLES_CONFIG.keys())
    for i, role in enumerate(roles):
        print(f"{i+1}. {role}")

    while True:
        try:
            choice = int(input(f"enter the number (1-{len(roles)}): "))
            if 1 <= choice <= len(roles):
                return roles[choice - 1]
            else:
                print("invalid number. try again")
        except ValueError:
            print("Invalid inpiut, please enter a number.")

def main_interview_loop(candidate_profile, interview_plan):
    session_results = []

    question_queue = []

    for skill_item in interview_plan['skill_distribution']:
        for _ in range(skill_item['count']):
            question_queue.append(skill_item)

    random.shuffle(question_queue)

    total_questions = len(question_queue)

    for i, question_details in enumerate(question_queue):
        print("\n" + "="*60)
        print(f"QUESTION {i+1} OF {total_questions}")
        print("="*60)

        # Generate the question
        interview_material = generate_question_and_benchmark(
            question_details['skill'],
            question_details['topic'],
            question_details['difficulty'],
            candidate_profile
        )

        if not interview_material:
            print("Could not generate question. SKipping.")
            continue

        print("\nINTERVIEW QUESTION:")
        print(interview_material['question'])

        # Get Candidate's answer
        candidate_answer = input("\nYour Answer: ")

        print("\nAnalyzing your answer, please wait...")

        # Evaluate Answer
        evaluation = evaluate_answer(candidate_answer, interview_material)

        # Store the result
        result = {
            "question_details": question_details,
            "question_text": interview_material['question'],
            "rubric": interview_material['rubric'],
            "candidate_answer": candidate_answer,
            "evaluation": evaluation
        }
        session_results.append(result)

        # print("\n--- Immediate Feedback ---")
        # print(f"Score: {evaluation['earned_points']} / {evaluation['total_points']}")
        # print(evaluation['feedback'])

    return session_results

def generate_final_report(results, candidate_profile, role):
    """GEnerates and prints the final summary report"""
    print("\n\n" + "#"*60)
    print("     FINAL INTERVIEW REPORT")
    print("#"*60)
    print(f"Candidate: \t{candidate_profile.get('name', 'N/A')}")
    print(f"Role: \t\t{role}")

    total_earned = sum(r['evaluation']['earned_points'] for r in results)
    
    total_possible = sum(r['evaluation']['total_points'] for r in results)

    if total_possible > 0:
        percentage = (total_earned / total_possible) * 100
        print(f"\nOverall Score: {total_earned} / {total_possible} ({percentage:.2f}%)")

    print("\n--- DETAILED BREAKDOWN ---")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Questoin (Skill: {result['question_details']['skill']})")
        print(f"   '{result['question_text']}'")
        print(f"   Score: {result['evaluation']['earned_points']} / {result['evaluation']['total_points']}")

        print("   --- Scoring Criteria ---")
        for rubric_item in result['rubric']:
            print(f"     - {rubric_item['concept']} ({rubric_item['points']} pts)")

        print("   --- Feedback Provided ---")
        # Indent the feedback for readability
        for line in result['evaluation']['feedback'].split('\n'):
            print(f"     {line}")

if __name__ == "__main__":
    resume_path = "Aditya_Lavaniya_Resume.pdf"

    try:
        reader = PdfReader(resume_path)
        resume_text = "".join(page.extract_text() for page in reader.pages)
    except FileNotFoundError:
        print(f"Error: The resume file was not found at '{resume_path}'")
        exit()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        exit()

    candidate_profile = parse_resume(resume_text)

    if not candidate_profile:
        exit()
    
    # MODIFICATION: Interactive role selection
    selected_role = select_role()

    interview_plan = setup_interview_configuration(selected_role, candidate_profile)

    if not interview_plan:
        exit()

    print("\nSetup complete. The interview is ready to begin.")
    input("Press Enter to start the first question...")

    final_results = main_interview_loop(candidate_profile, interview_plan)

    generate_final_report(final_results, candidate_profile, selected_role)