   
   
import requests
import streamlit as st
import logging  # Import the logging module

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def get_ollama_response(input_text):
    logging.info(f"response: {input_text}")
    response = requests.post(
        "http://localhost:8000/bot/invoke",
        json={'input': {'input': input_text}}
    )
    logging.info("I am here")
    if response.status_code != 200:
            logging.error(f"Request failed with status: {response.status_code}")
            return f"Error: {response.status_code}. {response.json().get('detail', 'No additional information')}"
    response_json = response.json().get('answer')
    return response_json

# Streamlit framework for the frontend
st.title('Chatbot using Langchain, Ollama and OpenAI API')

# Get user inputs
input_text = st.text_input("Ask a question about the SQuAD dataset context")

# Show response for input_text
if input_text:
    with st.spinner("Generating response..."):
        st.write(get_ollama_response(input_text))