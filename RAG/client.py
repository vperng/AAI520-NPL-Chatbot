import streamlit as st
import requests
from PIL import Image
import logging 
import base64
from io import BytesIO

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

# Set up the Streamlit app configuration
st.set_page_config(page_title="Chatbot Application", layout="centered")

# Custom CSS to style the background and text elements
st.markdown(
    """
    <style>
    div {
        background-color: lightblue;
    }
    
    h1 {
        color: darkred; /* Change title color */
        text-align: center; /* Center the title */
    }
    .stTextInput div[role="textbox"] > div::after {
        background: none !important;
    """,
    unsafe_allow_html=True
)
# border:1px solid black;
logo = Image.open("usd-logo.png")  

# Convert image to base64 for embedding
buffered = BytesIO()
logo.save(buffered, format="PNG")
logo_base64 = base64.b64encode(buffered.getvalue()).decode()

# Use custom HTML for side-by-side alignment with base64 image
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 75px; margin-right: 10px;">
        <h1 style="margin: 0;">SQUAD Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)



# Display the title of the chatbot app with custom color
st.write("Ask me anything, and I'll try my best to answer!")

# get chat response
def get_ollama_response(input_text):
    logging.info(f"response: {input_text}")
    response = requests.post(
        "https://karthikraghav-squaddockerapi.hf.space/",
        json={'input': {'input': input_text}}
    )
    logging.info("I am here")
    if response.status_code != 200:
            logging.error(f"Request failed with status: {response.status_code}")
            return f"Error: {response.status_code}. {response.json().get('detail', 'No additional information')}"
    response_json = response.json().get('answer')
    return response_json

# Get user inputs
input_text = st.text_input("Ask a question")

# Show response for input_text
if input_text:
    with st.spinner("Generating response..."):
        st.write(get_ollama_response(input_text))