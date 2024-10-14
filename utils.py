# utils.py

import os
import openai
import nltk
from nltk.tokenize import sent_tokenize
from unstructured.partition.auto import partition
import chromadb
from chromadb.config import Settings

# Download 'punkt' tokenizer data for NLTK
nltk.download('punkt', quiet=True)

# Load the OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_documents_in_folder(folder_path):
    """
    Processes all documents in the specified folder and its subfolders.

    Returns:
    - texts (list of str): List of text chunks from all documents.
    - metadata (list of dict): Metadata for each text chunk.
    """
    texts = []
    metadata = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                text_chunks = process_document(file_path)
                texts.extend([chunk['text'] for chunk in text_chunks])
                metadata.extend([{'source': file_path, **chunk['metadata']} for chunk in text_chunks])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return texts, metadata

def process_document(file_path):
    """
    Processes the document and extracts text content, splitting it into manageable chunks.

    Returns:
    - list of dict: List containing text chunks and their metadata.
    """
    elements = partition(filename=file_path)
    text = "\n\n".join([str(el) for el in elements])
    return chunk_text(text, file_path)

def chunk_text(text, file_path, max_tokens=500):
    """
    Splits the text into chunks of approximately max_tokens tokens.

    Returns:
    - list of dict: List containing text chunks and their metadata.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        if current_tokens + tokens <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += tokens
        else:
            chunks.append({'text': current_chunk.strip(), 'metadata': {'file_path': file_path}})
            current_chunk = sentence
            current_tokens = tokens
    if current_chunk:
        chunks.append({'text': current_chunk.strip(), 'metadata': {'file_path': file_path}})
    return chunks

def initialize_vector_database(texts, metadata):
    """
    Initializes or updates the vector database with the new embeddings.
    """
    # Initialize Chroma client with the new API
    client = chromadb.PersistentClient(path=".chromadb")

    # Create or get collection
    collection_name = "documents"
    # Remove existing collection to avoid duplicates
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.create_collection(name=collection_name)

    # Generate embeddings and add to collection
    embeddings = get_embeddings(texts)
    ids = [str(i) for i in range(len(texts))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadata,
        documents=texts
    )

def load_vector_database():
    """
    Loads the existing vector database.

    Returns:
    - collection: The Chroma collection instance, or None if not found.
    """
    try:
        client = chromadb.PersistentClient(path=".chromadb")
        collection = client.get_collection(name="documents")
        return collection
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

def get_embeddings(texts):
    """
    Generates embeddings for a list of texts using OpenAI's embeddings API.

    Returns:
    - list: List of embeddings.
    """
    embeddings = []
    batch_size = 1000  # Adjust based on your API rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        batch_embeddings = [data['embedding'] for data in response['data']]
        embeddings.extend(batch_embeddings)
    return embeddings

def generate_response(query, collection):
    """
    Generates a response for a given query by retrieving relevant context from the vector database.

    Returns:
    - response (str): The assistant's response.
    - source_metadata (list): Metadata of the sources used.
    """
    # Get embedding for the query
    query_embedding = get_embeddings([query])[0]

    # Search in the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
    )

    # Retrieve the most relevant contexts and metadata
    retrieved_texts = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    # Check if the query matches any document closely
    threshold = 0.5  # Adjust the threshold based on your needs
    if min(distances) > threshold:
        # The query is not similar to any document
        # Check if it's a common knowledge question
        general_response = answer_general_question(query)
        return general_response, []

    # Prepare context with references
    context = ""
    source_metadata = []
    for text, metadata in zip(retrieved_texts, retrieved_metadatas):
        context += f"Source: {metadata['source']}\n{text}\n\n---\n\n"
        source_metadata.append(metadata)

    # Prepare messages for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that answers questions strictly based on the provided context. If the answer is not in the context, respond that the information is not available."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        }
    ]

    # Generate response using ChatCompletion
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.0,
        )
        return completion.choices[0].message['content'].strip(), source_metadata
    except openai.error.OpenAIError as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time.", []

def answer_general_question(query):
    """
    Generates an answer for a general knowledge question.

    Returns:
    - response (str): The assistant's response indicating the information is from general knowledge.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that answers general knowledge questions. Indicate that this information is from general knowledge and not the provided documents."
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nAnswer:"
        }
    ]
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return completion.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."
