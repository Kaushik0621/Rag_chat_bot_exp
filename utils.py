import os
import numpy as np
import openai
import faiss
from unstructured.partition.auto import partition
import nltk
from nltk.tokenize import sent_tokenize

# Download 'punkt' tokenizer data for NLTK
nltk.download('punkt')

# Load the OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_document(file_path):
    """
    Processes the document and extracts text content, splitting it into manageable chunks.
    
    Parameters:
    - file_path (str): Path to the document file.
    
    Returns:
    - list of str: List of text chunks from the document.
    """
    elements = partition(filename=file_path)
    text = "\n\n".join([str(el) for el in elements])
    return chunk_text(text)

def chunk_text(text, max_tokens=500):
    """
    Splits the text into chunks of approximately max_tokens tokens.
    
    Parameters:
    - text (str): The text to split.
    - max_tokens (int): Approximate maximum number of tokens per chunk.
    
    Returns:
    - list of str: List of text chunks.
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
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(texts):
    """
    Generates embeddings for a list of texts using OpenAI's embeddings API.
    
    Parameters:
    - texts (list of str): List of texts to generate embeddings for.
    
    Returns:
    - list: List of embeddings, where each embedding is a list of floats.
    """
    embeddings = []
    batch_size = 1000  # Adjust based on your API rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        embeddings.extend([data['embedding'] for data in response['data']])
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from the provided embeddings.
    
    Parameters:
    - embeddings (list of list of floats): List of embeddings to index.
    
    Returns:
    - faiss.IndexFlatL2: A FAISS index of the embeddings.
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def generate_response(query, faiss_index, texts):
    """
    Generates a response for a given query by retrieving relevant context from the indexed documents and 
    sending it to OpenAI's GPT-3.5-turbo model for generating an answer.
    
    Parameters:
    - query (str): User's query.
    - faiss_index (faiss.IndexFlatL2): FAISS index containing the document embeddings.
    - texts (list of str): List of original document texts corresponding to embeddings.
    
    Returns:
    - str: The generated response from the OpenAI GPT-3.5-turbo model.
    """
    # Get embedding for the query
    query_embedding = get_embeddings([query])[0]
    # Search in the FAISS index
    D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k=5)
    # Retrieve the most relevant contexts
    context = "\n\n".join([texts[i] for i in I[0]])

    # Prepare messages for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that answers questions based on the provided context. If you don't know the exact answer, provide the closest possible answer."
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
            max_tokens=150,
            temperature=0.7,
        )
        return completion.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."
