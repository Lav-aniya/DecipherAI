import json
import copy

# JOB_ROLES_CONFIG = {
#     "Senior Python Developer": {
#         "total_questions": 5,
#         "skill_distribution": [
#             {"skill": "Python", "topic": "Advanced Concepts (e.g., Decorators, GIL)", "count": 2, "difficulty": "Advanced"},
#             {"skill": "System Design", "topic": "Scalability and Architecture", "count": 1, "difficulty": "Intermediate"},
#             {"skill": "Cloud (AWS)", "topic": "Practical Application (e.g., S3, Lambda)", "count": 1, "difficulty": "Intermediate"},
#             {"skill": "Behavioral", "topic": "Leadership and Teamwork", "count": 1, "difficulty": "N/A"}
#         ]
#     },
#     "Junior Python Developer": {
#         "total_questions": 4,
#         "skill_distribution": [
#             {"skill": "Python", "topic": "Core Fundamentals (e.g., Data Structures)", "count": 2, "difficulty": "Beginner"},
#             {"skill": "SQL", "topic": "Basic Queries (e.g., Joins)", "count": 1, "difficulty": "Beginner"},
#             {"skill": "Problem-Solving", "topic": "Algorithmic Thinking", "count": 1, "difficulty": "Beginner"}
#         ]
#     },
#     "Machine Learning Engineer": {
#         "total_questions": 7,
#         "skill_distribution": [
#             {"skill": "Machine Learning", "topic": "Core Fundamentals (e.g., Algorithms)", "count": 2, "difficulty": "Beginner"},
#             {"skill": "Deep Learning", "topic": "differet architectures (e.g., CNN, Transformers)", "count": 3, "difficulty": "Beginner"},
#             {"skill": "Mathematics", "topic": "Linear Algebra, Probability, Statistics, Calculus related to AI", "count": 2, "difficulty": "Beginner"}
#         ]
#     }
# }

JOB_ROLES_CONFIG = {
    "Senior Python Developer": {
        # The total_questions key is removed as it can be calculated from the counts below.
        "skill_distribution": [
            {
                "skill": "Python",
                "difficulty": "Advanced",
                "count": 2,
                # <-- NEW: A list of specific concepts for more targeted questions.
                "concepts": [
                    "The Global Interpreter Lock (GIL): its impact and workarounds",
                    "Advanced decorator patterns (e.g., with arguments, class decorators)",
                    "Metaclasses and their practical applications",
                    "Asynchronous programming with asyncio",
                    "Memory management and garbage collection in CPython"
                ]
            },
            {
                "skill": "System Design",
                "difficulty": "Intermediate",
                "count": 1,
                 # <-- NEW: The AI will be instructed to ask about a project from the candidate's resume.
                "topic": "PROBE_PROJECT_SKILL"
            },
            {
                "skill": "Cloud (AWS)",
                "difficulty": "Intermediate",
                "count": 1,
                "concepts": [
                    "Serverless architecture using AWS Lambda and API Gateway",
                    "Cost optimization strategies for S3 storage classes",
                    "Using Infrastructure as Code (IaC) with AWS CDK or CloudFormation",
                    "IAM roles vs. users for application permissions"
                ]
            },
            {
                "skill": "Behavioral",
                "difficulty": "N/A",
                "count": 1,
                # <-- NEW: Provides a framework for the question.
                "topic": "Describe a time you had a major technical disagreement with a colleague.",
                "framework": "STAR"
            }
        ]
    },
    "Junior Python Developer": {
        "skill_distribution": [
            {
                "skill": "Python",
                "difficulty": "Beginner",
                "count": 2,
                "concepts": [
                    "Difference between lists and tuples and when to use each",
                    "How Python dictionaries are implemented (hash maps)",
                    "What are list comprehensions and their benefits",
                    "Basics of object-oriented programming (Classes and Inheritance)"
                ]
            },
            {
                "skill": "SQL",
                "difficulty": "Beginner",
                "count": 1,
                "concepts": ["INNER vs. LEFT JOIN", "GROUP BY clause", "Primary Keys vs. Foreign Keys"]
            },
            {
                "skill": "Problem-Solving",
                "difficulty": "Beginner",
                "count": 1,
                "topic": "Algorithmic Thinking (e.g., reversing a string, finding max value in a list)"
            }
        ]
    },
    "Machine Learning Engineer": {
        "skill_distribution": [
            {
                "skill": "Machine Learning",
                "difficulty": "Intermediate", # <-- Adjusted difficulty
                "count": 4,
                "concepts": [
                    "Bias-Variance Tradeoff", "Cross-Validation Techniques",
                    "Regularization (L1 vs. L2)", "How Gradient Boosting works"
                ]
            },
            {
                "skill": "Deep Learning",
                "difficulty": "Intermediate", # <-- Adjusted difficulty
                "count": 4,
                # Corrected typo and made concepts more specific
                "concepts": [
                    "Role of activation functions in a neural network",
                    "Explain the self-attention mechanism in Transformers",
                    "How to prevent overfitting in a deep neural network",
                    "Differences between a CNN and an RNN",
                    "PROBE_PROJECT_SKILL" # <-- Ask about a specific DL project on their resume
                ]
            },
            {
                "skill": "Mathematics for AI",
                "difficulty": "Intermediate", # <-- Adjusted difficulty
                "count": 4,
                "concepts": [
                    "The role of the chain rule in backpropagation",
                    "Dot products in the context of neural networks",
                    "Bayes' Theorem and its application in machine learning"
                ]
            },
            {
                "skill": "Resume Deep Dive",
                "difficulty": "N/A",
                "count": 4, # <-- Set how many resume questions you want
                "concepts": [
                    "PROBE_SECTION:projects",
                    "PROBE_SECTION:work_experience",
                    "PROBE_SECTION:blogs",
                    "PROBE_SECTION:hackathons"
                ]
            }
        ]
    }
}


# def setup_interview_configuration(role, candidate_profile):
#     """
#     Creates the interview plan based on the selected role.
#     In a more advanced system, it would also use the candidate_profile
#     to further tailor the plan.
#     """
#     print(f"--- Setting up interview configuration for role: {role}... ---")
#     if role in JOB_ROLES_CONFIG:
#         # For now, we directly return the config.
#         # An advanced version could modify the difficulty or topics
#         # based on the skills found in candidate_profile.
#         print("Configuration found. Interview plan created.")
#         return JOB_ROLES_CONFIG[role]
#     else:
#         print("Role not found in configuration.")
#         return None
    

def setup_interview_configuration(role, candidate_profile):
    """
    Creates a DYNAMIC interview plan by tailoring the base configuration
    using the candidate's specific resume profile.
    """
    print(f"--- Setting up interview configuration for role: {role}... ---")
    
    if role not in JOB_ROLES_CONFIG:
        print("Role not found in configuration.")
        return None

    # 1. Make a deep copy to avoid changing the original config
    tailored_plan = copy.deepcopy(JOB_ROLES_CONFIG[role])
    
    candidate_skills = [skill.lower() for skill in candidate_profile.get("skills", [])]
    
    print("--- Analyzing candidate profile to tailor interview plan... ---")
    
    # 2. The core tailoring logic
    for question_block in tailored_plan.get("skill_distribution", []):
        # Example 1: Make a generic topic more specific
        if question_block["skill"] == "Cloud (AWS)" and "terraform" in candidate_skills:
            print("  -> Found specific skill 'Terraform', updating topic.")
            question_block["topic"] = "PROBE_PROJECT_SKILL"
            question_block["skill"] = "Terraform"
            
        # Example 2: Adjust difficulty based on experience
        try:
            experience_years = int(candidate_profile.get("total_experience", "0").split()[0])
            if experience_years > 5 and question_block["difficulty"] == "Intermediate":
                print(f"  -> High experience detected. Upgrading '{question_block['skill']}' to Advanced.")
                question_block["difficulty"] = "Advanced"
        except (ValueError, IndexError):
            pass
            
    print("Configuration found. Tailored interview plan created.")
    return tailored_plan


# sample_role = "Machine Learning Engineer"
# sample_profile = {"name": "John Doe", "skills": ["Python", "AWS"]} # This profile is not used by the function yet

# # Call the function
# output = setup_interview_configuration(sample_role, sample_profile)

# # Print the output in a clean format
# print("\n--- FUNCTION OUTPUT ---")
# print(json.dumps(output, indent=2))