// Global state
let currentModel = null;
let uploadedImage = null;
let availableModels = [];
let mediaRecorder = null;
let audioChunks = [];
let recordedAudio = null;

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
const systemPrompt = document.getElementById('systemPrompt');
const recordBtn = document.getElementById('recordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const audioPreview = document.getElementById('audioPreview');

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
    modelSelect.addEventListener('change', () => {
        const selectedModel = modelSelect.value;
        if (selectedModel) {
            const model = availableModels.find(m => m.name === selectedModel);
            if (model && model.loaded) {
                currentModel = selectedModel;
            } else {
                currentModel = null; // Model not loaded
            }
        } else {
            currentModel = null;
        }
        updateModelInfo();
    });
    
    imageUpload.addEventListener('change', handleImageUpload);
    
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
    });
    
    tokensSlider.addEventListener('input', (e) => {
        tokensValue.textContent = e.target.value;
    });
    
    recordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);
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
        const isCurrentModel = currentModel === selectedModel;
        const statusText = model.loaded ? 'Loaded ✓' : 'Not loaded';
        const currentText = isCurrentModel ? ' (Current)' : '';
        
        modelInfo.innerHTML = `
            <p><strong>Name:</strong> ${model.name}${currentText}</p>
            <p><strong>Type:</strong> ${model.type}</p>
            <p><strong>Status:</strong> ${statusText}</p>
            <p><strong>Description:</strong> ${model.description}</p>
        `;
        
        // Show/hide image upload based on model type
        const isVisionModel = model.type === 'vision-language';
        document.querySelector('.file-upload').style.display = isVisionModel ? 'block' : 'none';
    } else {
        modelInfo.innerHTML = '<p>No model selected</p>';
        document.querySelector('.file-upload').style.display = 'none';
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
    
    try {
        const model = availableModels.find(m => m.name === currentModel);
        
        if (model && model.type === 'vision-language' && uploadedImage) {
            // Handle image analysis (non-streaming)
            const typingMsg = addMessage('assistant', '<span class="typing-indicator"></span>');
            
            const formData = new FormData();
            formData.append('model', currentModel);
            formData.append('file', uploadedImage);
            if (prompt) {
                formData.append('prompt', prompt);
            }
            
            const response = await fetch('/api/image/analyze', {
                method: 'POST',
                body: formData
            });
            
            typingMsg.remove();
            
            if (response.ok) {
                const result = await response.json();
                addMessage('assistant', result.response || result.error || 'No response');
                
                // Update metrics
                if (result.generation_time) {
                    document.getElementById('genTime').textContent = `${result.generation_time.toFixed(2)}s`;
                }
            } else {
                const error = await response.json();
                addMessage('assistant', `Error: ${error.detail || 'Unknown error'}`);
            }
        } else {
            // Handle text generation with streaming
            await sendStreamingMessage(prompt);
        }
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessage('assistant', 'Failed to get response. Please try again.');
    }
}

// Send streaming message
async function sendStreamingMessage(prompt) {
    // Create message container for streaming response
    const assistantMsg = addMessage('assistant', '');
    const msgContent = assistantMsg.querySelector('p');
    
    let fullResponse = '';
    let startTime = Date.now();
    let tokensGenerated = 0;
    
    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: currentModel,
                prompt: prompt,
                temperature: parseFloat(tempSlider.value),
                max_tokens: parseInt(tokensSlider.value),
                system_prompt: systemPrompt.value || undefined
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            msgContent.textContent = `Error: ${error.detail || 'Unknown error'}`;
            return;
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data.trim()) {
                        try {
                            const json = JSON.parse(data);
                            
                            if (json.type === 'token') {
                                fullResponse += json.token;
                                msgContent.textContent = fullResponse;
                                tokensGenerated = json.tokens_generated;
                                
                                // Update metrics in real-time
                                document.getElementById('tokensPerSec').textContent = json.tokens_per_second.toFixed(2);
                                document.getElementById('tokensGen').textContent = tokensGenerated;
                                
                                // Scroll to bottom
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            } else if (json.type === 'final') {
                                // Final metrics
                                document.getElementById('genTime').textContent = `${json.generation_time.toFixed(2)}s`;
                                document.getElementById('tokensPerSec').textContent = json.tokens_per_second.toFixed(2);
                                document.getElementById('tokensGen').textContent = json.tokens_generated;
                            } else if (json.type === 'error') {
                                msgContent.textContent = `Error: ${json.error}`;
                                break;
                            }
                        } catch (e) {
                            console.error('Failed to parse SSE data:', e, data);
                        }
                    }
                }
            }
        }
        
        // Process any remaining buffer
        if (buffer.trim() && buffer.startsWith('data: ')) {
            const data = buffer.slice(6);
            if (data.trim()) {
                try {
                    const json = JSON.parse(data);
                    if (json.type === 'token') {
                        fullResponse += json.token;
                        msgContent.textContent = fullResponse;
                    }
                } catch (e) {
                    console.error('Failed to parse final SSE data:', e, data);
                }
            }
        }
        
        if (!fullResponse) {
            msgContent.textContent = 'No response generated.';
        }
        
    } catch (error) {
        console.error('Streaming error:', error);
        msgContent.textContent = 'Failed to get streaming response. Please try again.';
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

// Audio recording functions
async function startRecording() {
    try {
        // Check if mediaDevices is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('MediaDevices API not supported in this browser');
        }
        
        // Check permissions first
        const permissions = await navigator.permissions.query({ name: 'microphone' });
        console.log('Microphone permission status:', permissions.state);
        
        if (permissions.state === 'denied') {
            throw new Error('Microphone permission denied. Please enable in browser settings.');
        }
        
        addMessage('system', 'Requesting microphone access...');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        // Check if we got a valid stream
        if (!stream || stream.getAudioTracks().length === 0) {
            throw new Error('No audio tracks available');
        }
        
        console.log('Audio stream obtained:', stream.getAudioTracks());
        
        // Try to use WAV format if available
        let mimeType = 'audio/webm;codecs=opus'; // Default
        
        // Check supported MIME types
        const possibleTypes = [
            'audio/wav',
            'audio/wave',
            'audio/mpeg',
            'audio/mp4',
            'audio/webm;codecs=opus',
            'audio/webm'
        ];
        
        for (const type of possibleTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
                console.log(`Using MIME type: ${type}`);
                mimeType = type;
                break;
            }
        }
        
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: mimeType
        });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            recordedAudio = audioBlob;
            console.log('Recorded audio blob type:', audioBlob.type);
            
            // Show audio player in preview
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPreview.innerHTML = `
                <p>Audio recorded (${(audioBlob.size / 1024).toFixed(1)} KB)</p>
                <audio controls>
                    <source src="${audioUrl}" type="audio/webm">
                    Your browser does not support the audio element.
                </audio>
                <br>
                <button id="transcribeBtn" class="btn btn-primary" style="margin-top: 10px;">📝 Transcribe to Text</button>
            `;
            
            // Add transcribe button event listener
            document.getElementById('transcribeBtn').addEventListener('click', transcribeAudio);
            
            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        
        // Update UI
        recordBtn.style.display = 'none';
        stopRecordBtn.style.display = 'inline-block';
        recordingStatus.textContent = '🔴 Recording...';
        recordingStatus.className = 'recording-status recording';
        
        addMessage('system', 'Recording started. Click "Stop Recording" when finished.');
        
    } catch (error) {
        console.error('Detailed error accessing microphone:', error);
        
        let errorMessage = 'Error: Could not access microphone. ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Permission denied. Please click the microphone icon in the address bar and allow access.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found. Please check your hardware.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage += 'Browser does not support audio recording.';
        } else if (error.name === 'NotReadableError') {
            errorMessage += 'Microphone is in use by another application.';
        } else {
            errorMessage += error.message;
        }
        
        addMessage('system', errorMessage);
        
        // Show debug info
        addMessage('system', `Debug info: ${error.name} - ${error.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        // Update UI
        recordBtn.style.display = 'inline-block';
        stopRecordBtn.style.display = 'none';
        recordingStatus.textContent = '✅ Recording completed';
        recordingStatus.className = 'recording-status completed';
        
        addMessage('system', 'Recording stopped. You can now transcribe the audio or start a new recording.');
    }
}

async function transcribeAudio() {
    if (!recordedAudio) {
        addMessage('system', 'No audio recording found.');
        return;
    }
    
    const transcribeBtn = document.getElementById('transcribeBtn');
    transcribeBtn.disabled = true;
    transcribeBtn.textContent = '⏳ Transcribing...';
    
    try {
        console.log('Starting transcription...');
        console.log('Audio blob size:', recordedAudio.size, 'bytes');
        console.log('Audio blob type:', recordedAudio.type);
        
        const formData = new FormData();
        formData.append('file', recordedAudio, 'recording.webm');
        
        addMessage('system', `Sending ${(recordedAudio.size / 1024).toFixed(1)} KB audio for transcription...`);
        
        const response = await fetch('/api/audio/transcribe', {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        
        if (response.ok) {
            const result = await response.json();
            console.log('Transcription result:', result);
            
            if (result.error) {
                addMessage('system', `Transcription error: ${result.error}`);
                return;
            }
            
            const transcription = result.transcription || 'No transcription available';
            
            // Insert transcription into chat input
            chatInput.value = transcription;
            
            addMessage('system', `Audio transcribed: "${transcription}"`);
            
            // Update audio preview to show transcription
            audioPreview.innerHTML += `
                <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                    <strong>Transcription:</strong><br>
                    "${transcription}"
                </div>
            `;
            
            if (result.transcription_time) {
                addMessage('system', `Transcription took ${result.transcription_time.toFixed(2)} seconds`);
            }
        } else {
            const errorText = await response.text();
            console.error('Server error response:', errorText);
            
            try {
                const error = JSON.parse(errorText);
                addMessage('system', `Transcription failed: ${error.detail || 'Unknown error'}`);
            } catch {
                addMessage('system', `Transcription failed: ${errorText}`);
            }
        }
    } catch (error) {
        console.error('Transcription error:', error);
        addMessage('system', `Failed to transcribe audio: ${error.message}`);
    } finally {
        transcribeBtn.disabled = false;
        transcribeBtn.textContent = '📝 Transcribe to Text';
    }
}