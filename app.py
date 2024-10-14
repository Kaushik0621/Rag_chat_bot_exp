import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Initialize MongoDB Client
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['mydatabase']
collection = db['document_embeddings']

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2Templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Initialize ChatOpenAI model with GPT-4
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Fetch top relevant documents based on query vector
def get_relevant_documents(query_vector, top_k=5):
    documents = list(collection.find())
    doc_embeddings = [doc['embedding'] for doc in documents]
    doc_texts = [doc['text'] for doc in documents]
    doc_metadata = [doc['metadata'] for doc in documents]

    # Compute cosine similarity between query vector and document embeddings
    similarities = cosine_similarity([query_vector], doc_embeddings).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Retrieve top relevant documents
    relevant_docs = [{"text": doc_texts[i], "metadata": doc_metadata[i]} for i in top_indices]
    return relevant_docs

@app.post("/chat")
async def chat(question_request: QuestionRequest):
    question = question_request.question

    try:
        # Generate embedding for the user's question
        query_vector = embeddings_model.embed_query(question)

        # Retrieve relevant documents
        relevant_docs = get_relevant_documents(query_vector)

        # Prepare context for the GPT-4 model
        context = "\n".join([doc['text'] for doc in relevant_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        # Get answer from GPT-4 model
        response = llm.invoke(prompt)

        # Access the content of the response directly
        answer = response.content

        # Prepare references for display
        references = [
            {
                "source": doc["metadata"].get("source", "Unknown"),
                "page": doc["metadata"].get("page", "N/A"),
                "text": doc["text"][:200]
            }
            for doc in relevant_docs
        ]

        return JSONResponse({
            "answer": answer,
            "references": references
        })

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": "Unable to retrieve response."}, status_code=500)

@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
