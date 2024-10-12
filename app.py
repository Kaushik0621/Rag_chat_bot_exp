import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import threading
from utils import process_document, get_embeddings, create_faiss_index, generate_response

app = Flask(__name__)
load_dotenv()

# Global variables to store texts and the FAISS index
texts = []
faiss_index = None
lock = threading.Lock()
is_processing = False  # Track processing status

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global texts, faiss_index, is_processing
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'status': 'error', 'message': 'No files uploaded.'}), 400

    uploaded_filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join("temp", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        uploaded_filenames.append(file_path)

    # Set processing to True and start document processing
    is_processing = True
    threading.Thread(target=process_and_index_documents, args=(uploaded_filenames,)).start()

    return jsonify({'status': 'success', 'message': 'Files uploaded and processing started.'})

def process_and_index_documents(file_paths):
    global texts, faiss_index, is_processing, lock
    with lock:
        new_texts = []
        for file_path in file_paths:
            text_chunks = process_document(file_path)
            new_texts.extend(text_chunks)
        texts.extend(new_texts)

        embeddings = get_embeddings(texts)
        faiss_index = create_faiss_index(embeddings)

    is_processing = False  # Set processing to False when done

@app.route('/chat', methods=['POST'])
def chat():
    global texts, faiss_index, is_processing
    data = request.get_json()
    query = data.get('message', '')
    if not query:
        return jsonify({'status': 'error', 'message': 'No message provided.'}), 400

    if is_processing:
        return jsonify({'status': 'error', 'message': 'Documents are still processing. Please wait a moment and try again.'}), 400

    if faiss_index is None:
        return jsonify({'status': 'error', 'message': 'No documents have been processed yet.'}), 400

    response = generate_response(query, faiss_index, texts)
    return jsonify({'status': 'success', 'response': response})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)
