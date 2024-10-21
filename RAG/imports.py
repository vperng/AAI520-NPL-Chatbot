from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.load import dumpd, dumps, load, loads
from langchain.chains import load_chain
from langserve import add_routes
from langchain_core.runnables import RunnableBinding, RunnableLambda
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import pandas as pd
import json
import os
import json
import torch
import numpy as np
import logging  
import sqlite3
import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO
import threading


