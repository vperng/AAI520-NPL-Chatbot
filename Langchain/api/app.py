from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langserve import add_routes
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import pandas as pd
import json
import os
import numpy as np
import logging  # Import the logging module

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# Get the absolute path of the current file
env_path = Path.cwd().parent / ".env"

# Load the .env file
load_dotenv(dotenv_path=env_path)

logging.info("Loading data into Langchain.")


def squad1_json_to_dataframe(file_path, record_path=['data', 'paragraphs', 'qas', 'answers']):
    """
    Functuon to convert the dataset JSON file to a Pandas DataFrame.

    file_path (str): Path to the JSON file
    record_path (list): Path to the deepest level in the JSON structure (default is ['data', 'paragraphs', 'qas', 'answers']).

    Returns dataFrame containing the parsed data.
    """
    # Load JSON data
    with open(file_path, 'r') as f:
        file_data = json.load(f)

    # Extract and normalize the nested JSON structures
    answers_df = pd.json_normalize(file_data, record_path)
    questions_df = pd.json_normalize(file_data, record_path[:-1])
    paragraphs_df = pd.json_normalize(file_data, record_path[:-2])

    # Create 'context' by repeating the corresponding paragraph for each question
    questions_df['context'] = np.repeat(paragraphs_df['context'].values, paragraphs_df.qas.str.len())

    # Create final DataFrame with necessary columns
    data = questions_df[['id', 'question', 'context', 'answers']].copy()

    # Extract text and start positions to separate column
    data['answer_text'] = data['answers'].apply(lambda x: x[0]['text'] if len(x) > 0 else "")
    data['answer_start'] = data['answers'].apply(lambda x: x[0]['answer_start'] if len(x) > 0 else None)

    # Add 'c_id' to uniquely identify each context
    data['c_id'] = pd.factorize(data['context'])[0]

    return data.reset_index(drop=True)

# Load the SQuAD dataset
file_path = os.getenv("SQUAD_DATASET_PATH", "../../train-v1.1.json")
df = squad1_json_to_dataframe(file_path, record_path=['data', 'paragraphs', 'qas', 'answers'])
df_context = pd.DataFrame(df['context'].unique(), columns=['context'])

# Load data into Langchain
loader = DataFrameLoader(df_context, page_content_column="context")
docs = loader.load()

logging.info("split documents into chunks.")
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
logging.info(documents[:5])
logging.info("Creating embedding.")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever()

# Define the LLM and prompt template
llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context from Stanford Question Answer database. 
Think step by step before providing a detailed answer. 
<context>
{context} 
</context>
Question: {input}""")

logging.info(prompt)

logging.info("creating retrieval chains.")

document_chain=create_stuff_documents_chain(llm, prompt)

# Create the retrieval-based document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Define the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Chatbot API Server"
)


@app.post("/bot/invoke")
async def get_response(input: dict):
    try:
        logging.info(f'got a request: {input}')

        query = (input.get('input')).get('input')
        logging.info(f"Received query: {query}")
        logging.info(retrieval_chain)
        response = retrieval_chain.invoke({"input": query})
        logging.info(response['answer'])
        if response:
            return {"answer": response['answer']}
        else:
            return {"answer": "No answer generated. Please try with a different query."}
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.info("Starting FastAPI server at http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)





