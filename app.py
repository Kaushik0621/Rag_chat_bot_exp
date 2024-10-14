from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Set up the template directory
templates = Jinja2Templates(directory="templates")

# Serve the index.html template
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Initialize embeddings and load the persisted vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db'
)

# Set up the retriever
retriever = vectorstore.as_retriever()

# Initialize the language model
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Define the prompt template for the question-answering
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain using document retrieval and the defined prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get('question', '')

    # Generate answer using the retrieval chain
    result = qa_chain.invoke({"input": question})

    # Debugging: Print the entire result to understand its structure
    print("Result:", result)  # Check server logs for result structure

    # Extract the answer, checking for possible key names
    answer = result.get("output") or result.get("result") or result.get("answer") or "I'm sorry, I couldn't find an answer."
    source_docs = result.get("source_documents", [])

    # Prepare references
    references = []
    for doc in source_docs:
        ref = {
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 'N/A'),
            'paragraph': doc.metadata.get('paragraph', 'N/A'),
            'snippet': doc.page_content[:200]  # Snippet from the document
        }
        references.append(ref)

    return {"answer": answer, "references": references}
