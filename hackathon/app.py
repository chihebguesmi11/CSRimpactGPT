from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai
from PDFEmbeddingManager import PDFEmbeddingManager

app = Flask(__name__)
CORS(app)

load_dotenv()

# API key definition
openai.api_key = "my key"
PDF_DIRECTORY = "data"

# Initialize embedding manager
embedding_manager = PDFEmbeddingManager(openai.api_key, 
                                    PDF_DIRECTORY, 
                                    embedding_file="embeddings.pkl",
                                    existing_actions_embedding_file="existing_actions_embeddings.pkl")

class ConversationManager:
    VALID_STATES = {'initial', 'details', 'carbon_footprint', 'investment', 'complete', 'followup'}
    VALID_OPTIONS = {'sustainability', 'social impact'}

    @staticmethod
    def validate_state(state):
        return state in ConversationManager.VALID_STATES

    @staticmethod
    def validate_option(option):
        return option.lower() in ConversationManager.VALID_OPTIONS

    @staticmethod
    def get_expensya_info():
        # Query the embedding database for Expensya information
        query = "What is Expensya's industry and business domain?"
        similar_docs = embedding_manager.search(query)
        return similar_docs[0] if similar_docs else None

    @staticmethod
    def get_initial_prompt():
        return (
            "Welcome to Expensya's Sustainability and Social Impact Portal!\n"
            "Please choose one of the following options:\n"
            "1. Sustainability\n"
            "2. Social Impact\n"
            "Type your choice to proceed."
        )

class StrategyGenerator:
    @staticmethod
    def generate_sustainability_prompt(data):
        expensya_info = ConversationManager.get_expensya_info()
        industry_context = (
            f"Context: Expensya is a leading expense management SaaS company in the {data.get('industry', 'fintech')} industry, "
            "specializing in automated expense management and digital transformation."
        )
        
        return (
            f"You are a sustainability consultant specialized in carbon footprint reduction for SaaS companies. "
            f"{industry_context}\n"
            f"Current situation:\n"
            f"- Focus area: {data['goal_details']}\n"
            f"- Current carbon footprint: {data['carbon_footprint']}\n"
            f"- Available investment: {data['investment']}\n\n"
            f"Please provide:\n"
            f"1. A detailed 12-month strategy with quarterly milestones specifically tailored for Expensya's digital services\n"
            f"2. Specific technological solutions and their estimated impact, focusing on cloud infrastructure and digital processes\n"
            f"3. ROI calculations and environmental impact metrics for a SaaS business model\n"
            f"4. Risk assessment and mitigation strategies for a fintech company\n\n"
            f"Include specific recommendations for:\n"
            f"- Green hosting solutions for SaaS infrastructure\n"
            f"- Energy-efficient data centers\n"
            f"- Sustainable development practices\n"
            f"After providing the strategy, ask if they would like any clarifications or have specific aspects they'd like to explore further."
        )

    @staticmethod
    def generate_social_impact_prompt(data):
        expensya_info = ConversationManager.get_expensya_info()
        industry_context = (
            f"Context: Expensya is a leading expense management SaaS company in the {data.get('industry', 'fintech')} industry, "
            "focused on digital transformation and process automation."
        )
        
        return (
            f"You are a social impact consultant specialized in sustainable development for fintech companies. "
            f"{industry_context}\n"
            f"Project details:\n"
            f"- Focus area: {data['goal_details']}\n"
            f"- Available investment: {data['investment']}\n\n"
            f"Please provide:\n"
            f"1. A detailed 12-month implementation plan tailored for Expensya's digital service environment\n"
            f"2. Impact measurement framework specific to fintech operations\n"
            f"3. Stakeholder engagement strategy focusing on both digital and traditional banking sectors\n"
            f"4. Risk assessment and success metrics for SaaS implementation\n\n"
            f"Include specific recommendations for:\n"
            f"- Digital inclusion initiatives\n"
            f"- Financial literacy programs\n"
            f"- Sustainable fintech practices\n"
            f"After providing the strategy, ask if they would like any clarifications or have specific aspects they'd like to explore further."
        )

    @staticmethod
    def generate_followup_prompt(previous_context, user_question):
        return (
            f"Previous context: {previous_context}\n\n"
            f"User follow-up question: {user_question}\n\n"
            f"Please provide a detailed response to the user's question, maintaining consistency with the previous strategy "
            f"and considering Expensya's context as a fintech company. Be specific and actionable in your recommendations."
        )

def generate_final_strategy(data, user_option):
    try:
        if user_option == "sustainability":
            prompt = StrategyGenerator.generate_sustainability_prompt(data)
        else:
            prompt = StrategyGenerator.generate_social_impact_prompt(data)

        # Use embedding manager to generate response using both databases
        similar_docs = embedding_manager.search(prompt)
        filtered_docs = embedding_manager.filter_recommendations_by_existing_actions(similar_docs)
        
        # Generate response using filtered documents
        enhanced_response = ""
        if filtered_docs:
            context = "\n".join([doc['content'] for doc in filtered_docs])
            messages = [
                {"role": "system", "content": f"Context from Expensya knowledge base:\n{context}\n\nUse this context to help generate a response to:\n{prompt}"},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            enhanced_response = response['choices'][0]['message']['content']
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            enhanced_response = response['choices'][0]['message']['content']
        
        data["previous_strategy"] = enhanced_response
        
        # Debugging: Log the strategy text to see if it's correct
        app.logger.info(f"Generated Strategy: {enhanced_response}")
        
        # Generate roadmap from the strategy
        roadmap = generate_roadmap(enhanced_response)
        
        return jsonify({
            "response": enhanced_response,
            "roadmap": roadmap,  # Include the roadmap in the response
            "conversation_state": "complete",
            "additional_data": data
        })

    except Exception as e:
        app.logger.error(f"Error generating strategy: {str(e)}")
        return jsonify({
            "response": "I'm having trouble generating your strategy. Please try again.",
            "conversation_state": "initial"
        })



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format"}), 400
            
        user_input = data.get("message", "").strip()
        conversation_state = data.get("conversation_state", "initial")
        user_option = data.get("user_option", "")
        additional_data = data.get("additional_data", {})

        if conversation_state == "complete":
            return handle_followup_question(user_input, additional_data, user_option)

        if not ConversationManager.validate_state(conversation_state):
            return jsonify({"error": "Invalid conversation state"}), 400

        if conversation_state == "initial":
            return handle_initial_state(user_input)
            
        return handle_conversation_state(conversation_state, user_input, user_option, additional_data)

    except Exception as e:
        app.logger.error(f"Error in generate_response: {str(e)}")
        return jsonify({
            "response": "I encountered a temporary issue. Could you please rephrase your question?",
            "conversation_state": "complete"
        })

def handle_initial_state(user_input):
    if not user_input:
        return jsonify({
            "response": ConversationManager.get_initial_prompt(),
            "conversation_state": "initial"
        })

    if "sustainability" in user_input.lower():
        return jsonify({
            "response": "What specific aspect of sustainability would you like to focus on for Expensya? (e.g., reducing cloud infrastructure emissions, sustainable office practices, paperless initiatives)",
            "conversation_state": "details",
            "user_option": "sustainability"
        })
    elif "social impact" in user_input.lower():
        return jsonify({
            "response": "What specific area of social impact would you like to focus on for Expensya? (e.g., financial inclusion, digital literacy, employee well-being)",
            "conversation_state": "details",
            "user_option": "social impact"
        })
    
    return jsonify({
        "response": "Please specify if you're interested in 'sustainability' or 'social impact'",
        "conversation_state": "initial"
    })

def handle_conversation_state(state, user_input, user_option, additional_data):
    if state == "details":
        additional_data["goal_details"] = user_input
        # Get Expensya's industry from embeddings
        expensya_info = ConversationManager.get_expensya_info()
        if expensya_info:
            additional_data["industry"] = "fintech"  # Set default industry
        
        if user_option == "sustainability":
            return jsonify({
                "response": "What is Expensya's current annual carbon footprint? (in metric tons of CO2)",
                "conversation_state": "carbon_footprint",
                "user_option": user_option,
                "additional_data": additional_data
            })
        return jsonify({
            "response": "What is the planned investment budget for this social impact initiative? (in USD)",
            "conversation_state": "investment",
            "user_option": user_option,
            "additional_data": additional_data
        })

    elif state == "carbon_footprint":
        additional_data["carbon_footprint"] = user_input
        return jsonify({
            "response": "What is the planned investment budget for sustainability initiatives? (in USD)",
            "conversation_state": "investment",
            "user_option": user_option,
            "additional_data": additional_data
        })

    elif state == "investment":
        additional_data["investment"] = user_input
        return generate_final_strategy(additional_data, user_option)

    return jsonify({"error": "Invalid state transition"}), 400

def handle_followup_question(user_input, additional_data, user_option):
    try:
        previous_strategy = additional_data.get("previous_strategy", "")
        prompt = StrategyGenerator.generate_followup_prompt(previous_strategy, user_input)
        
        # Use embedding manager for followup questions
        similar_docs = embedding_manager.search(prompt)
        filtered_docs = embedding_manager.filter_recommendations_by_existing_actions(similar_docs)
        
        if filtered_docs:
            context = "\n".join([doc['content'] for doc in filtered_docs])
            messages = [
                {"role": "system", "content": f"Context from Expensya knowledge base:\n{context}\n\nUse this context to help generate a response to:\n{prompt}"},
                {"role": "user", "content": user_input}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
            )
        
        followup_response = response['choices'][0]['message']['content']
        additional_data["previous_strategy"] = f"{previous_strategy}\n\nFollow-up: {user_input}\nResponse: {followup_response}"
        
        return jsonify({
            "response": followup_response,
            "conversation_state": "complete",
            "additional_data": additional_data
        })
    except Exception as e:
        app.logger.error(f"Error in followup: {str(e)}")
        return jsonify({
            "response": "I'm having trouble understanding. Could you rephrase your question?",
            "conversation_state": "complete"
        })
    
def generate_roadmap(strategy_text):
    try:
        # Define the prompt
        prompt = (
            "Summarize the following sustainability strategy into a clear 12-month roadmap. "
            "Organize the roadmap into quarters with specific actions, key technologies, "
            "and measurable goals:\n\n"
            f"{strategy_text}"
        )

        # Log the generated prompt
        app.logger.info(f"Roadmap Generation Prompt: {prompt}")

        # Generate the response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in sustainability strategy development."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the generated roadmap
        roadmap = response["choices"][0]["message"]["content"]
        
        # Log the generated roadmap
        app.logger.info(f"Generated Roadmap: {roadmap}")
        
        return roadmap

    except Exception as e:
        app.logger.error(f"Error generating roadmap: {str(e)}")
        return "Error generating roadmap. Please try again."
        


if __name__ == '__main__':
    # Process existing actions PDF on startup
    """embedding_manager.process_existing_actions_pdf("action_data/Ive-heard-of-Expensya.pdf")"""
    app.run(debug=True)