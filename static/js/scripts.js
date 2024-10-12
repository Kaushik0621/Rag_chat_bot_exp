document.addEventListener('DOMContentLoaded', function() {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const chatSection = document.getElementById('chat-section');
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    uploadButton.addEventListener('click', function() {
        const files = fileInput.files;
        if (files.length === 0) {
            alert('Please select files to upload.');
            return;
        }

        uploadStatus.textContent = 'Uploading and processing files...';
        const formData = new FormData();
        for (let file of files) {
            formData.append('files[]', file);
        }

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                uploadStatus.textContent = 'Files uploaded and processing started.';
                chatSection.style.display = 'block';
            } else {
                uploadStatus.textContent = 'Error: ' + data.message;
            }
        })
        .catch(error => {
            uploadStatus.textContent = 'Error uploading files.';
            console.error(error);
        });
    });

    sendButton.addEventListener('click', function() {
        const message = userInput.value.trim();
        if (message === '') {
            alert('Please enter a message.');
            return;
        }

        appendMessage('You', message, 'user-message');
        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                appendMessage('Bot', data.response, 'bot-message');
            } else {
                appendMessage('Bot', 'Error: ' + data.message, 'bot-message');
            }
        })
        .catch(error => {
            appendMessage('Bot', 'Error fetching response.', 'bot-message');
            console.error(error);
        });
    });

    function appendMessage(sender, message, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', className);
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
