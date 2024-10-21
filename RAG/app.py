from imports import *

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

 # Set up the Streamlit app configuration
st.set_page_config(page_title="Chatbot Application", layout="centered")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
   
# Initialize the SentenceTransformer model
# model_name = 'all-MiniLM-L6-v2'
# sentence_transformer_model = SentenceTransformer(model_name).to(device)

# Initialize the SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

#sentence_transformer_model = SentenceTransformer(model_name).to(device)

# Wrap the SentenceTransformer model with LangChain's HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = Chroma(persist_directory="./", embedding_function=embeddings, collection_name="squadembedding")

logging.info("initialize retriever")
retriever = db.as_retriever()


# Define the LLM and prompt template
logging.info("initialize model and prompt.")

llm = Ollama(model="llama2") 

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context} 
</context>
Question: {input}""")

logging.info(prompt)

# Create retrieval chains
logging.info("creating retrieval chains")
document_chain=create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

logging.info("initialize FAST api.")

# Define the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Chatbot API Server"
)

@app.get("/")
async def read_root():
    return {"message": f"Welcome to SQUAD chatbot"}


@app.post("/")
async def get_response(input: dict):
    try:
        logging.info(f'got a request: {input}')

        query = (input.get('input')).get('input')
        logging.info(f"Received query: {query}")
        conversation_history = (input.get('history', []))  # Fetch conversation history

        # Combine history and the current query
        full_context = "\n".join(conversation_history + [query])
        logging.info(f"Received query: {full_context}")

        response = retrieval_chain.invoke({"input": full_context})
        logging.info(response['answer'])
        if response:
            return {"answer": response['answer']}
        else:
            return {"answer": "No answer generated. Please try with a different query."}
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Function to run FastAPI
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit frontend
st.title("Squad Chatbot application")

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
   
# get chat response
def get_ollama_response(input_text):
    logging.info(f"response: {input_text}")
    response = requests.post(
        "http://localhost:8000/",
        json={
                'input': {'input': input_text},
                'history': st.session_state.history
            }
    )
    logging.info("I am here")
    if response.status_code != 200:
            logging.error(f"Request failed with status: {response.status_code}")
            return f"Error: {response.status_code}. {response.json().get('detail', 'No additional information')}"
    
    return response

# Get user inputs
input_text = st.text_input("Ask a question")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Append current user input to the conversation history
st.session_state.history.append(f"User: {input_text}")

# Show response for input_text
if input_text:
    with st.spinner("Generating response..."):
        response = get_ollama_response(input_text)
        answer_text = response.json().get('answer')
        answer_text = answer_text.replace('Based on the provided context,','')
        st.session_state.history.append(f"Bot: {answer_text}")  # Store the bot's response
        st.write(answer_text)

# Show the entire conversation
if st.session_state.history:
    st.write("SQUAD Conversation:")
    for message in st.session_state.history:
        st.write(message)











