let selectedFile = null;
let totalCost = 0;
let currentRequestId = null;

// DOM elements
const chatContainer = document.getElementById('chatContainer');
const inputForm = document.getElementById('inputForm');
const textInput = document.getElementById('textInput');
const fileInput = document.getElementById('fileInput');
const fileButton = document.getElementById('fileButton');
const fileName = document.getElementById('fileName');
const sendButton = document.getElementById('sendButton');
const totalCostDisplay = document.getElementById('totalCost');

// Event listeners
fileButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
inputForm.addEventListener('submit', handleSubmit);
textInput.addEventListener('input', autoResize);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        fileName.textContent = file.name;
    }
}

function autoResize() {
    textInput.style.height = 'auto';
    textInput.style.height = textInput.scrollHeight + 'px';
}

async function handleSubmit(e) {
    e.preventDefault();
    
    const text = textInput.value.trim();
    
    if (!text && !selectedFile) {
        return;
    }
    
    // Disable input
    sendButton.disabled = true;
    
    // Add user message
    if (text) {
        addMessage('user', text, selectedFile ? `📎 ${selectedFile.name}` : null);
    } else if (selectedFile) {
        addMessage('user', `📎 Uploaded: ${selectedFile.name}`);
    }
    
    // Clear input
    textInput.value = '';
    textInput.style.height = 'auto';
    
    // Show loading
    const loadingId = addMessage('agent', '<div class="loading"></div>');
    
    try {
        // Prepare form data
        const formData = new FormData();
        if (text) formData.append('text', text);
        if (selectedFile) formData.append('file', selectedFile);
        if (currentRequestId) {
            formData.append('previous_request_id', currentRequestId);
            formData.append('clarification_response', text);
        }
        
        // Send request
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Handle response
        handleResponse(data);
        
        // Clear file selection
        selectedFile = null;
        fileName.textContent = '';
        fileInput.value = '';
        currentRequestId = null;
        
    } catch (error) {
        removeMessage(loadingId);
        addMessage('agent', `❌ Error: ${error.message}`);
    } finally {
        sendButton.disabled = false;
        textInput.focus();
    }
}

function handleResponse(data) {
    if (data.status === 'needs_clarification') {
        // Need clarification
        currentRequestId = data.request_id;
        addMessage('system', `🤔 ${data.clarification_question}`);
    } else if (data.status === 'completed') {
        // Success
        displayResult(data);
        
        // Update cost
        if (data.total_cost) {
            totalCost += data.total_cost;
            totalCostDisplay.textContent = totalCost.toFixed(4);
        }
    } else if (data.status === 'failed') {
        // Error
        addMessage('agent', `❌ Error: ${data.error_message}`);
    }
}

function displayResult(data) {
    let content = '';
    
    // Show extracted content if available
    if (data.extracted_content && data.extracted_content.text) {
        const excerpt = data.extracted_content.text.length > 500 
            ? data.extracted_content.text.substring(0, 500) + '...'
            : data.extracted_content.text;
        
        content += `<div class="result-section">
            <h4>📄 Extracted Content:</h4>
            <div class="extracted-content">${escapeHtml(excerpt)}</div>
            <div class="metadata">
                Confidence: ${(data.extracted_content.confidence * 100).toFixed(0)}% | 
                Method: ${data.extracted_content.extraction_method}
            </div>
        </div>`;
    }
    
    // Show result based on task type
    if (data.result) {
        content += formatTaskResult(data.result);
    }
    
    // Show execution info
    if (data.execution_plan) {
        content += `<div class="metadata">
            Task: ${data.execution_plan.task_type} | 
            Time: ${data.result?.execution_time_seconds || 0}s | 
            Cost: $${data.total_cost.toFixed(4)}
        </div>`;
    }
    
    addMessage('agent', content);
}

function formatTaskResult(result) {
    const output = result.output;
    let html = '<div class="result-section"><h4>✅ Result:</h4>';
    
    switch (result.task_type) {
        case 'summarization':
            html += `
                <p><strong>One-line summary:</strong><br>${output.one_line}</p>
                <p><strong>Key points:</strong></p>
                <ul>
                    ${output.bullets.map(b => `<li>${b}</li>`).join('')}
                </ul>
                <p><strong>Detailed summary:</strong><br>${output.five_sentence}</p>
            `;
            break;
            
        case 'sentiment_analysis':
            html += `
                <p><strong>Sentiment:</strong> ${output.label.toUpperCase()} 
                   (${(output.confidence * 100).toFixed(0)}% confident)</p>
                <p><strong>Justification:</strong> ${output.justification}</p>
            `;
            break;
            
        case 'code_explanation':
            html += `
                <p><strong>Language:</strong> ${output.language}</p>
                <p><strong>Explanation:</strong><br>${output.explanation}</p>
                ${output.potential_bugs.length > 0 ? `
                    <p><strong>⚠️ Potential Issues:</strong></p>
                    <ul>${output.potential_bugs.map(b => `<li>${b}</li>`).join('')}</ul>
                ` : ''}
                <p><strong>Complexity:</strong> Time: ${output.time_complexity}, Space: ${output.space_complexity}</p>
            `;
            break;
            
        case 'text_extraction':
        case 'youtube_transcript':
            html += `<p>Successfully extracted ${output.word_count} words.</p>`;
            break;
            
        case 'conversational':
            html += `<p>${output.response}</p>`;
            break;
            
        default:
            html += `<pre>${JSON.stringify(output, null, 2)}</pre>`;
    }
    
    html += '</div>';
    return html;
}

function addMessage(type, content, subtitle = null) {
    const messageId = 'msg-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.id = messageId;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (subtitle) {
        contentDiv.innerHTML = `<div style="font-size: 11px; opacity: 0.8; margin-bottom: 5px;">${subtitle}</div>${content}`;
    } else {
        contentDiv.innerHTML = content;
    }
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageId;
}

function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-focus on text input
textInput.focus();