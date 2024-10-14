import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from pymongo import MongoClient
import nltk

# Ensure nltk Punkt tokenizer is downloaded
nltk.download('punkt')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['mydatabase']
collection = db['document_embeddings']

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load documents from the 'documents' directory
print("Loading documents...")
documents = []
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

# Split documents and generate embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

for i, doc in enumerate(docs):
    text = doc.page_content
    metadata = doc.metadata
    embedding = embeddings_model.embed_query(text)

    # Prepare the data to store in MongoDB
    doc_data = {
        "text": text,
        "embedding": embedding,  # Vector as an array
        "metadata": metadata
    }

    # Store document data in MongoDB
    collection.insert_one(doc_data)

print("Document processing and storage completed.")
