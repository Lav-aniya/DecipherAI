
JOB_ROLES_CONFIG = {
    "Senior Python Developer": {
        "total_questions": 5,
        "skill_distribution": [
            {"skill": "Python", "topic": "Advanced Concepts (e.g., Decorators, GIL)", "count": 2, "difficulty": "Advanced"},
            {"skill": "System Design", "topic": "Scalability and Architecture", "count": 1, "difficulty": "Intermediate"},
            {"skill": "Cloud (AWS)", "topic": "Practical Application (e.g., S3, Lambda)", "count": 1, "difficulty": "Intermediate"},
            {"skill": "Behavioral", "topic": "Leadership and Teamwork", "count": 1, "difficulty": "N/A"}
        ]
    },
    "Junior Python Developer": {
        "total_questions": 4,
        "skill_distribution": [
            {"skill": "Python", "topic": "Core Fundamentals (e.g., Data Structures)", "count": 2, "difficulty": "Beginner"},
            {"skill": "SQL", "topic": "Basic Queries (e.g., Joins)", "count": 1, "difficulty": "Beginner"},
            {"skill": "Problem-Solving", "topic": "Algorithmic Thinking", "count": 1, "difficulty": "Beginner"}
        ]
    },
    "Machine Learning Engineer": {
        "total_questions": 7,
        "skill_distribution": [
            {"skill": "Machine Learning", "topic": "Core Fundamentals (e.g., Algorithms)", "count": 2, "difficulty": "Beginner"},
            {"skill": "Deep Learning", "topic": "differet architectures (e.g., CNN, Transformers)", "count": 3, "difficulty": "Beginner"},
            {"skill": "Mathematics", "topic": "Linear Algebra, Probability, Statistics, Calculus related to AI", "count": 2, "difficulty": "Beginner"}
        ]
    }
}

def setup_interview_configuration(role, candidate_profile):
    """
    Creates the interview plan based on the selected role.
    In a more advanced system, it would also use the candidate_profile
    to further tailor the plan.
    """
    print(f"--- Setting up interview configuration for role: {role}... ---")
    if role in JOB_ROLES_CONFIG:
        # For now, we directly return the config.
        # An advanced version could modify the difficulty or topics
        # based on the skills found in candidate_profile.
        print("Configuration found. Interview plan created.")
        return JOB_ROLES_CONFIG[role]
    else:
        print("Role not found in configuration.")
        return None