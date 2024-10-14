import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
import nltk

# Ensure nltk Punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print("Loading documents...")
# Initialize a list to hold all documents
documents = []

# Directory containing documents
docs_dir = 'documents'

# Walk through the directory and load supported files
for root, dirs, files in os.walk(docs_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if file.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        elif file.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"Unsupported file type: {file}")
            continue

        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")
# Split documents into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# Add metadata to documents
for i, doc in enumerate(docs):
    doc.metadata['chunk'] = i + 1

print("Initializing OpenAI embeddings...")
# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

print("Creating vector store using Chroma...")
# Create vector store using Chroma
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory='chroma_db'  # Directory where Chroma will store the database
)
print("Vector store created.")

print("Persisting the vector store...")
vectorstore.persist()
print("Process completed successfully.")
