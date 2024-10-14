document.addEventListener('DOMContentLoaded', function() {
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const processingStatus = document.getElementById('processing-status');
    const chatSection = document.getElementById('chat-section');
    const documentHolder = document.getElementById('document-holder');

    // Function to check processing status
    // ... existing code ...

function checkProcessingStatus() {
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: 'status_check' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'ready') {
            processingStatus.style.display = 'none';
            chatSection.style.display = 'block';
        } else if (data.status === 'not_ready') {
            processingStatus.textContent = 'Vector database not found. Please run "process_documents.py" first.';
        } else {
            processingStatus.textContent = 'Processing...';
            setTimeout(checkProcessingStatus, 5000);
        }
    })
    .catch(error => {
        processingStatus.textContent = 'Error checking status.';
        console.error(error);
    });
}

checkProcessingStatus();

// ... existing code ...


    checkProcessingStatus();

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
                displaySourceDocuments(data.metadata);
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

    function displaySourceDocuments(metadataList) {
        documentHolder.innerHTML = '';
        if (metadataList && metadataList.length > 0) {
            const header = document.createElement('h3');
            header.textContent = 'Source Documents';
            documentHolder.appendChild(header);

            metadataList.forEach(metadata => {
                const docDiv = document.createElement('div');
                docDiv.classList.add('source-document');
                docDiv.innerHTML = `<p><strong>Source:</strong> ${metadata.source}</p>`;
                documentHolder.appendChild(docDiv);
            });
        }
    }
});
