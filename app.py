import gradio as gr
import time
import random
from groq import Groq
from prof import get_resume_text, parse_resume
from interview_config import setup_interview_configuration, JOB_ROLES_CONFIG
from core import generate_question_and_benchmark, evaluate_answer

def generate_report_markdown(results, candidate_profile, role):
    """
    Generates a formatted Markdown string for the final report.
    This is a modified version of our previous print function.
    """
    report_parts = []
    report_parts.append(f"# Final Interview Report\n")
    report_parts.append(f"**Candidate:** {candidate_profile.get('name', 'N/A')}  \n")
    report_parts.append(f"**Role:** {role}\n")
    
    total_earned = sum(r['evaluation']['earned_points'] for r in results)
    total_possible = sum(r['evaluation']['total_points'] for r in results)
    
    if total_possible > 0:
        percentage = (total_earned / total_possible) * 100
        report_parts.append(f"## Overall Score: {total_earned} / {total_possible} ({percentage:.2f}%)\n")
    
    report_parts.append("\n---\n ## Detailed Breakdown\n")
    
    for i, result in enumerate(results):
        report_parts.append(f"### Question {i+1} (Skill: {result['question_details']['skill']})\n")
        report_parts.append(f"> {result['question_text']}\n")
        report_parts.append(f"**Score:** {result['evaluation']['earned_points']} / {result['evaluation']['total_points']}\n")
        
        report_parts.append("**Scoring Criteria:**\n")
        for rubric_item in result['rubric']:
            report_parts.append(f"- *{rubric_item['concept']}* ({rubric_item['points']} pts)")
        
        report_parts.append("\n**Feedback Provided:**\n")
        # Ensure feedback is formatted nicely in Markdown
        feedback_md = result['evaluation']['feedback'].replace("✅", "\n✅").replace("🤔", "\n🤔")
        report_parts.append(feedback_md)
        report_parts.append("\n---\n")
        
    return "".join(report_parts)

audio_client = Groq()

def transcribe_audio(audio_path):
    if not audio_path:
        return ""
    print("-- transcribing audio --")
    with open(audio_path, "rb") as file:
        transcription = audio_client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo"
        )
    print(f"Transcription: {transcription.text}")
    return transcription.text

def generate_speech(text, file_path="ai_response.wav"):
    print("-- generating audio --")
    response = audio_client.audio.speech.create(
        model="playai-tts",
        voice="Fritz-PlayAI",
        input=text,
        response_format="wav"
    )
    response.write_to_file(file_path)
    return file_path

with gr.Blocks(theme=gr.themes.Soft(), title="DecipherAI") as demo:

    # State management
    session_state = gr.State({
        "candidate_profile": None,
        "interview_plan": None,
        "question_queue": [],
        "session_results": [],
        "current_question_material": None,
        "current_question_details": None
    })

    # UI LAYOUT

    #setup screen
    with gr.Column(visible=True) as setup_screen:
        gr.Markdown("DecipherAI Technical Interviewer")
        gr.Markdown("Upload your resume, select a role, and click 'Start Interview' ")
        with gr.Row():
            resume_file = gr.File(label="Upload Resume (PDF)")
            role_dropdown = gr.Dropdown(choices=list(JOB_ROLES_CONFIG.keys()), label="Select Role")
        start_button = gr.Button("Start Interview", variant="primary")
        status_box = gr.Markdown("")

    #interview screen (initially hidden)
    with gr.Column(visible=False) as interview_screen:
        chatbot = gr.Chatbot(label="Interview Chat", height=500, type="tuples")
        text_input = gr.Textbox(label="Your Answer", placeholder="Type your answer here ..")
        ai_audio_output = gr.Audio(label="Interviewer", autoplay=True, visible=True)
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Answer")
        submit_button = gr.Button("Submit Answer", variant="primary")

    #Report Screen (Initially hidden)
    with gr.Column(visible=False) as report_screen:
        final_report_display = gr.Markdown()
        restart_button = gr.Button("Start New Interview")

    # Event handlers: functions that run on user interaction

    def start_interview(file, role, current_state):
        """
        Triggered by the 'Start Interview' button
        parses resume, sets up the plan, and transitions the UI
        """

        if file is None or role is None:
            return {status_box: gr.update(value="Please upload a resume and select a role")}
        
        yield {status_box: gr.update(value="*Reading PDF...*")}
        resume_text = get_resume_text(file.name)
        if not resume_text:
            return {status_box: gr.update(value="Error reading PDF file")}
        
        yield {status_box: gr.update(value="*Parsing resume..*")}
        candidate_profile = parse_resume(resume_text)
        if not candidate_profile:
            return {status_box: gr.update(value="Error parsing resume")}
        current_state["candidate_profile"] = candidate_profile

        yield {status_box: gr.update(value="*Creating tailored interview plan...*")}
        interview_plan = setup_interview_configuration(role, candidate_profile)
        if not interview_plan:
            return {status_box: gr.update(value="Error creating interview plan")}
        current_state["interview_plan"] = interview_plan

        current_state["interview_plan"]["role_name"] = role

        # Prepare the question queue
        # question_queue = []
        # for skill_item in interview_plan['skill_distribution']:
        #     for _ in range(skill_item['count']):
        #         question_queue.append(skill_item)
        # random.shuffle(question_queue)

        question_queue = []
        for skill_item in interview_plan['skill_distribution']:
        # Check if the item has a list of concepts
            if "concepts" in skill_item:
            # If so, pick one concept randomly for each question count
                for _ in range(skill_item['count']):
                    topic = random.choice(skill_item["concepts"])
                    # Create a new, clean dictionary for the queue
                    new_question = {
                        "skill": skill_item["skill"],
                        "difficulty": skill_item["difficulty"],
                        "topic": topic
                    }
                    question_queue.append(new_question)
            else:
            # If it just has a single topic, add it normally
                for _ in range(skill_item['count']):
                    question_queue.append(skill_item)

        #random.shuffle(question_queue)

        current_state["question_queue"] = question_queue


        # Generate the first question
        first_question_details = current_state["question_queue"].pop(0)
        current_state["current_question_details"] = first_question_details
        first_question_material = generate_question_and_benchmark(
            first_question_details['skill'], first_question_details['topic'],
            first_question_details['difficulty'], candidate_profile
        )
        current_state["current_question_material"] = first_question_material


        # Initial chat message
        initial_chat = [(None, f"Hello! Let's begin. Your first question is:\n\n" + first_question_material['question'])]
        
        # UI Transition
        yield {
            setup_screen: gr.update(visible=False),
            interview_screen: gr.update(visible=True),
            chatbot: gr.update(value=initial_chat),
            session_state: current_state
        }
        
    def chat_turn(text_input, audio_input, chat_history, current_state):
        """
        Handles a single turn of the interview.
        Accepts EITHER text or audio input.
        Produces BOTH text and audio output.
        """
        
        user_answer = ""
        if text_input:
            user_answer = text_input
        elif audio_input:
            user_answer = transcribe_audio(audio_input)
        
        # if not user_answer: # Handle empty submission
        #     # Just return the current state without doing anything
        #     yield {
        #         chatbot: chat_history, 
        #         text_input: "", 
        #         audio_input: None, 
        #         ai_audio_output: None
        #     }
        #     return
        
        if not user_answer:
            yield [
                gr.update(), 
                gr.update(), 
                chat_history, 
                "", 
                None, 
                gr.update(), 
                None, 
                gr.update()
            ]
            return

        
        chat_history.append((user_answer, None))
        # Clear both inputs and show "thinking"
        # yield {
        #     chatbot: chat_history, 
        #     text_input: "", 
        #     audio_input: None, 
        #     ai_audio_output: None
        # }
        yield [
            gr.update(), 
            gr.update(), 
            chat_history, 
            "", 
            None, 
            gr.update(), 
            None, 
            gr.update()
        ]

        
        chat_history.append((None, "*Analyzing your answer...*"))
        #yield {chatbot: chat_history}
        yield [
            gr.update(), 
            gr.update(), 
            chat_history, 
            gr.update(), 
            gr.update(), 
            gr.update(), 
            gr.update(), 
            gr.update()
        ]
        time.sleep(1) 

        evaluation = evaluate_answer(user_answer, current_state["current_question_material"])

        # Store the full result
        result = {
            "question_details": current_state["current_question_details"],
            "question_text": current_state["current_question_material"]['question'],
            "rubric": current_state["current_question_material"]['rubric'],
            "candidate_answer": user_answer,
            "evaluation": evaluation
        }
        current_state["session_results"].append(result)

        
        if not current_state["question_queue"]:
            final_report_md = generate_report_markdown(
                current_state["session_results"], 
                current_state["candidate_profile"], 
                current_state["interview_plan"].get('role_name', 'Selected Role')
            )
            
            ai_text_response = "The interview is now complete. Here is your final report."
            ai_audio_path = generate_speech(ai_text_response)
            
            # yield {
            #     interview_screen: gr.update(visible=False),
            #     report_screen: gr.update(visible=True),
            #     final_report_display: gr.update(value=final_report_md),
            #     ai_audio_output: gr.update(value=ai_audio_path, visible=True)
            # }
            yield [
                gr.update(visible=False), # interview_screen
                gr.update(visible=True),  # report_screen
                gr.update(),              # chatbot
                gr.update(),              # text_input
                gr.update(),              # audio_input
                gr.update(value=final_report_md), # final_report_display
                gr.update(value=ai_audio_path, visible=True), # ai_audio_output
                gr.update()               # session_state
            ]
            return # End the function

        
        next_question_details = current_state["question_queue"].pop(0)
        current_state["current_question_details"] = next_question_details
        next_question_material = generate_question_and_benchmark(
            next_question_details['skill'], 
            next_question_details['topic'],
            next_question_details['difficulty'], 
            current_state["candidate_profile"]
        )
        current_state["current_question_material"] = next_question_material
        
        ai_text_response = next_question_material['question']
        
        
        ai_audio_path = generate_speech(ai_text_response)
        
        chat_history.append((None, ai_text_response))
        # yield {
        #     chatbot: chat_history,
        #     session_state: current_state,
        #     ai_audio_output: gr.update(value=ai_audio_path, visible=True)
        # }
        yield [
            gr.update(), # interview_screen
            gr.update(), # report_screen
            chat_history, # chatbot
            gr.update(), # text_input
            gr.update(), # audio_input
            gr.update(), # final_report_display
            gr.update(value=ai_audio_path, visible=True), # ai_audio_output
            current_state # session_state
        ]

    # def chat_turn(user_answer, chat_history, current_state):
    #     """
    #     Triggered by submitting an answer.
    #     Evaluates the answer and presents the next question or the final report.
    #     """

    #     # Append user's answer to chat
    #     chat_history.append((user_answer, None))
    #     yield {chatbot: chat_history, answer_box: gr.update(value="", interactive=False)}

    #     # Show a "thinking" message
    #     chat_history.append((None, "*Analyzing your answer...*"))
    #     yield {chatbot: chat_history}
    #     time.sleep(1) # Small delay for UX

    #     # Evaluate the previous answer
    #     evaluation = evaluate_answer(user_answer, current_state["current_question_material"])

    #     # Store the full result
    #     result = {
    #         "question_details": current_state["current_question_details"],
    #         "question_text": current_state["current_question_material"]['question'],
    #         "rubric": current_state["current_question_material"]['rubric'],
    #         "candidate_answer": user_answer,
    #         "evaluation": evaluation
    #     }
    #     current_state["session_results"].append(result)

    #     # Check if the interview is over
    #     if not current_state["question_queue"]:
    #         # Interview is over, generate report
    #         final_report_md = generate_report_markdown(
    #             current_state["session_results"], current_state["candidate_profile"], current_state["interview_plan"].get('role_name', 'Selected Role')
    #         )
    #         yield {
    #             interview_screen: gr.update(visible=False),
    #             report_screen: gr.update(visible=True),
    #             final_report_display: gr.update(value=final_report_md)
    #         }
    #     else:
    #         # There are more questions, generate the next one
    #         next_question_details = current_state["question_queue"].pop(0)
    #         current_state["current_question_details"] = next_question_details
    #         next_question_material = generate_question_and_benchmark(
    #             next_question_details['skill'], next_question_details['topic'],
    #             next_question_details['difficulty'], current_state["candidate_profile"]
    #         )
    #         current_state["current_question_material"] = next_question_material
            
    #         chat_history.append((None, next_question_material['question']))
    #         yield {
    #             chatbot: chat_history,
    #             answer_box: gr.update(interactive=True), # Re-enable textbox
    #             session_state: current_state
    #         }

    def restart_interview():
        """Resets the UI to the initial setup screen."""
        return {
            setup_screen: gr.update(visible=True),
            interview_screen: gr.update(visible=False),
            report_screen: gr.update(visible=False),
            chatbot: [],
            text_input: "",       # <-- Clear text input
            audio_input: None,    # <-- Clear audio input
            ai_audio_output: None,# <-- Clear audio output
            status_box: "",
            session_state: {
                "candidate_profile": None, "interview_plan": None, "question_queue": [],
                "session_results": [], "current_question_material": None, "current_question_details": None
            }
        }
    
    # def restart_interview():
    #     """Resets the UI to the initial setup screen."""
    #     return {
    #         setup_screen: gr.update(visible=True),
    #         interview_screen: gr.update(visible=False),
    #         report_screen: gr.update(visible=False),
    #         chatbot: [],
    #         answer_box: "",
    #         status_box: "",
    #         session_state: {
    #             "candidate_profile": None, "interview_plan": None, "question_queue": [],
    #             "session_results": [], "current_question_material": None, "current_question_details": None
    #         }
    #     }

    
    start_button.click(
        fn=start_interview,
        inputs=[resume_file, role_dropdown, session_state],
        outputs=[setup_screen, interview_screen, chatbot, status_box, session_state]
    )

    # Define the inputs and outputs for a chat turn
    # This makes it easy to reuse
    chat_inputs = [text_input, audio_input, chatbot, session_state]
    chat_outputs = [
        interview_screen, report_screen, chatbot, 
        text_input, audio_input,  # Clear both inputs
        final_report_display, ai_audio_output, # Update both outputs
        session_state
    ]
    
    text_input.submit(
        fn=chat_turn,
        inputs=chat_inputs,
        outputs=chat_outputs
    )

    audio_input.stop_recording(
        fn=chat_turn,
        inputs=chat_inputs,
        outputs=chat_outputs
    )

    # answer_box.submit(
    #     fn=chat_turn,
    #     inputs=[answer_box, chatbot, session_state],
    #     outputs=[interview_screen, report_screen, chatbot, answer_box, final_report_display, session_state]
    # )

    # submit_button.click(
    #     fn=chat_turn,
    #     inputs=[answer_box, chatbot, session_state],
    #     outputs=[interview_screen, report_screen, chatbot, answer_box, final_report_display, session_state]
    # )
    
    restart_button.click(
        fn=restart_interview,
        inputs=None,
        outputs=[
            setup_screen, 
            interview_screen, 
            report_screen, 
            chatbot, 
            text_input,         # <-- Must be in outputs
            audio_input,        # <-- Must be in outputs
            ai_audio_output,    # <-- Must be in outputs
            status_box, 
            session_state
        ]
    )

if __name__ == "__main__":
    demo.launch()