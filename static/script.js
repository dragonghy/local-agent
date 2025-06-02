// Global state
let currentModel = null;
let uploadedImage = null;
let availableModels = [];

// DOM elements
const modelSelect = document.getElementById('modelSelect');
const loadModelBtn = document.getElementById('loadModelBtn');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const modelInfo = document.getElementById('modelInfo');
const tempSlider = document.getElementById('temperature');
const tempValue = document.getElementById('tempValue');
const tokensSlider = document.getElementById('maxTokens');
const tokensValue = document.getElementById('tokensValue');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadAvailableModels();
    setupEventListeners();
});

// Event listeners
function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    loadModelBtn.addEventListener('click', loadSelectedModel);
    modelSelect.addEventListener('change', updateModelInfo);
    
    imageUpload.addEventListener('change', handleImageUpload);
    
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
    });
    
    tokensSlider.addEventListener('input', (e) => {
        tokensValue.textContent = e.target.value;
    });
}

// Load available models
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        availableModels = await response.json();
        
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = `${model.name} (${model.type})`;
            if (model.loaded) {
                option.textContent += ' ✓';
                if (!currentModel) {
                    currentModel = model.name;
                    option.selected = true;
                }
            }
            modelSelect.appendChild(option);
        });
        
        updateModelInfo();
    } catch (error) {
        console.error('Failed to load models:', error);
        addMessage('system', 'Failed to load available models. Please refresh the page.');
    }
}

// Load selected model
async function loadSelectedModel() {
    const selectedModel = modelSelect.value;
    if (!selectedModel) return;
    
    loadModelBtn.disabled = true;
    loadModelBtn.textContent = 'Loading...';
    
    try {
        const response = await fetch(`/api/load_model/${selectedModel}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            currentModel = selectedModel;
            addMessage('system', `Model ${selectedModel} loaded successfully!`);
            await loadAvailableModels();
        } else {
            const error = await response.json();
            addMessage('system', `Failed to load model: ${error.detail}`);
        }
    } catch (error) {
        console.error('Failed to load model:', error);
        addMessage('system', 'Failed to load model. Please try again.');
    } finally {
        loadModelBtn.disabled = false;
        loadModelBtn.textContent = 'Load Model';
    }
}

// Update model info
function updateModelInfo() {
    const selectedModel = modelSelect.value;
    const model = availableModels.find(m => m.name === selectedModel);
    
    if (model) {
        modelInfo.innerHTML = `
            <p><strong>Name:</strong> ${model.name}</p>
            <p><strong>Type:</strong> ${model.type}</p>
            <p><strong>Status:</strong> ${model.loaded ? 'Loaded ✓' : 'Not loaded'}</p>
            <p><strong>Description:</strong> ${model.description}</p>
        `;
        
        // Show/hide image upload based on model type
        const isVisionModel = model.type === 'vision-language';
        document.querySelector('.file-upload').style.display = isVisionModel ? 'block' : 'none';
    } else {
        modelInfo.innerHTML = '<p>No model selected</p>';
    }
}

// Send message
async function sendMessage() {
    const prompt = chatInput.value.trim();
    if (!prompt) return;
    
    if (!currentModel) {
        addMessage('system', 'Please select and load a model first.');
        return;
    }
    
    // Add user message
    addMessage('user', prompt);
    chatInput.value = '';
    
    // Show typing indicator
    const typingMsg = addMessage('assistant', '<span class="typing-indicator"></span>');
    
    try {
        let response;
        const model = availableModels.find(m => m.name === currentModel);
        
        if (model && model.type === 'vision-language' && uploadedImage) {
            // Handle image analysis
            const formData = new FormData();
            formData.append('model', currentModel);
            formData.append('file', uploadedImage);
            if (prompt) {
                formData.append('prompt', prompt);
            }
            
            response = await fetch('/api/image/analyze', {
                method: 'POST',
                body: formData
            });
        } else {
            // Handle text generation
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: currentModel,
                    prompt: prompt,
                    temperature: parseFloat(tempSlider.value),
                    max_tokens: parseInt(tokensSlider.value)
                })
            });
        }
        
        typingMsg.remove();
        
        if (response.ok) {
            const result = await response.json();
            addMessage('assistant', result.response || result.error || 'No response');
            
            // Update metrics
            if (result.generation_time) {
                document.getElementById('genTime').textContent = `${result.generation_time.toFixed(2)}s`;
            }
            if (result.tokens_per_second) {
                document.getElementById('tokensPerSec').textContent = result.tokens_per_second.toFixed(2);
            }
            if (result.tokens_generated) {
                document.getElementById('tokensGen').textContent = result.tokens_generated;
            }
        } else {
            const error = await response.json();
            addMessage('assistant', `Error: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        typingMsg.remove();
        console.error('Failed to send message:', error);
        addMessage('assistant', 'Failed to get response. Please try again.');
    }
}

// Add message to chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    // Handle code blocks and formatting
    if (role === 'assistant' && content.includes('```')) {
        content = formatCodeBlocks(content);
    }
    
    messageDiv.innerHTML = `<p>${content}</p>`;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return messageDiv;
}

// Format code blocks
function formatCodeBlocks(content) {
    return content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
    });
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    uploadedImage = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `
            <p>Uploaded: ${file.name}</p>
            <img src="${e.target.result}" alt="Uploaded image">
        `;
    };
    reader.readAsDataURL(file);
    
    addMessage('system', `Image uploaded: ${file.name}`);
}

// WebSocket support (for future streaming)
function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/chat');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Handle streaming responses
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
    };
    
    return ws;
}