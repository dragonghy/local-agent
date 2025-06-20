// Global state
let mediaRecorder = null;
let audioChunks = [];
let recordedAudio = null;
let recordingStartTime = null;
let recordingTimer = null;
let transcriptionHistory = [];
let maxRecordingTime = 10 * 60 * 1000; // 10 minutes in milliseconds
let minRecordingTime = 1000; // 1 second minimum
let currentStream = null; // Store current media stream
let isCancelling = false; // Flag to prevent onstop handler during cancel

// DOM elements
const recordBtn = document.getElementById('recordBtn');
const cancelBtn = document.getElementById('cancelBtn');
const confirmBtn = document.getElementById('confirmBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTime = document.getElementById('recordingTime');
const audioPreview = document.getElementById('audioPreview');
const resultsSection = document.getElementById('resultsSection');
const transcriptionResults = document.getElementById('transcriptionResults');
const historyList = document.getElementById('historyList');
const historySearch = document.getElementById('historySearch');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
    checkHttpsRequirement();
});

// Collapsible model selection
function toggleModelSelection() {
    const content = document.getElementById('modelOptions');
    const icon = document.querySelector('.toggle-icon');
    
    if (content.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
        icon.classList.remove('collapsed');
    } else {
        content.classList.add('collapsed');
        icon.classList.add('collapsed');
    }
}

// Check HTTPS requirement and show notice if needed
function checkHttpsRequirement() {
    const httpsNotice = document.getElementById('httpsNotice');
    const recordBtn = document.getElementById('recordBtn');
    
    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        httpsNotice.style.display = 'block';
        recordBtn.disabled = true;
        recordBtn.textContent = '🔒 HTTPS Required for Recording';
        recordBtn.style.background = '#6c757d';
    }
}

// Event listeners
function setupEventListeners() {
    recordBtn.addEventListener('click', startRecording);
    cancelBtn.addEventListener('click', cancelRecording);
    confirmBtn.addEventListener('click', confirmAndTranscribe);
    clearHistoryBtn.addEventListener('click', clearHistory);
    historySearch.addEventListener('input', filterHistory);
}

// Audio recording functions
async function startRecording() {
    try {
        // Check browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('MediaDevices API not supported in this browser');
            return;
        }

        // Check if running on HTTPS (required for microphone access in most browsers)
        if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
            showError('Microphone access requires HTTPS. Please use https:// or run the server with --https flag');
            return;
        }

        // Request microphone access
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        const stream = currentStream;

        // Check supported MIME types
        let mimeType = 'audio/webm;codecs=opus';
        const possibleTypes = [
            'audio/wav',
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/mpeg'
        ];

        for (const type of possibleTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
                mimeType = type;
                break;
            }
        }

        // Initialize MediaRecorder
        mediaRecorder = new MediaRecorder(stream, { 
            mimeType,
            bitsPerSecond: 128000 // Better quality
        });
        audioChunks = [];
        recordingStartTime = Date.now();
        isCancelling = false; // Reset cancelling flag

        // Set up recording event handlers
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            // Don't process if we're cancelling
            if (isCancelling) {
                isCancelling = false; // Reset flag
                return;
            }
            
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            recordedAudio = audioBlob;
            
            // Create audio preview
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPreview.innerHTML = `
                <p>Audio recorded (${(audioBlob.size / 1024).toFixed(1)} KB)</p>
                <audio controls>
                    <source src="${audioUrl}" type="${mimeType}">
                    Your browser does not support the audio element.
                </audio>
            `;

            // Update UI to show confirm/cancel buttons
            updateRecordingUI('completed');

            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording with data collection every 100ms
        mediaRecorder.start(100);
        
        // Update UI
        updateRecordingUI('recording');
        
        // Start timer
        startRecordingTimer();
        
        // Auto-stop after max recording time
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                stopRecordingTimer();
                showError(`Recording automatically stopped after ${maxRecordingTime / 60000} minutes`);
            }
        }, maxRecordingTime);

    } catch (error) {
        console.error('Error starting recording:', error);
        let errorMessage = 'Could not access microphone. ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Permission denied. Please allow microphone access.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage += 'Browser does not support audio recording.';
        } else if (error.name === 'NotReadableError') {
            errorMessage += 'Microphone is in use by another application.';
        } else {
            errorMessage += error.message;
        }
        
        showError(errorMessage);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        stopRecordingTimer();
        updateRecordingUI('completed');
    }
}

function cancelRecording() {
    // Set cancelling flag to prevent onstop handler
    isCancelling = true;
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        // Stop the recorder
        mediaRecorder.stop();
        stopRecordingTimer();
    }
    
    // Stop all media tracks to release microphone
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    // Clear recorded audio
    recordedAudio = null;
    audioPreview.innerHTML = '<p>Recording cancelled</p>';
    
    // Reset UI to idle state immediately
    updateRecordingUI('idle');
    
    // Clear any existing timers
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
    
    // Reset recording start time
    recordingStartTime = null;
}

function updateRecordingUI(state) {
    switch (state) {
        case 'recording':
            recordBtn.style.display = 'none';
            cancelBtn.style.display = 'inline-block';
            confirmBtn.style.display = 'inline-block';
            recordingStatus.textContent = '🔴 Recording... Click Confirm when done';
            recordingStatus.className = 'recording-status recording';
            break;
            
        case 'completed':
            recordBtn.style.display = 'none';
            cancelBtn.style.display = 'inline-block';
            confirmBtn.style.display = 'inline-block';
            recordingStatus.textContent = '✅ Recording completed';
            recordingStatus.className = 'recording-status completed';
            break;
            
        case 'idle':
        default:
            recordBtn.style.display = 'inline-block';
            cancelBtn.style.display = 'none';
            confirmBtn.style.display = 'none';
            recordingStatus.textContent = '';
            recordingStatus.className = 'recording-status';
            recordingTime.textContent = '00:00';
            break;
    }
}

function startRecordingTimer() {
    recordingTimer = setInterval(() => {
        if (recordingStartTime) {
            const elapsed = Date.now() - recordingStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            recordingTime.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}

function stopRecordingTimer() {
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
}

// Transcription functions
async function confirmAndTranscribe() {
    // If still recording, check minimum time and stop
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        const elapsedTime = Date.now() - recordingStartTime;
        
        if (elapsedTime < minRecordingTime) {
            showError(`Please record for at least ${minRecordingTime/1000} second(s)`);
            return;
        }
        
        return new Promise((resolve) => {
            // Set up a one-time listener for when recording stops
            const originalOnStop = mediaRecorder.onstop;
            mediaRecorder.onstop = () => {
                // Call the original onstop handler
                originalOnStop();
                // Then proceed with transcription
                setTimeout(() => {
                    proceedWithTranscription();
                    resolve();
                }, 200);
            };
            
            mediaRecorder.stop();
            stopRecordingTimer();
        });
    } else {
        proceedWithTranscription();
    }
}

async function proceedWithTranscription() {
    if (!recordedAudio) {
        showError('No audio recorded');
        return;
    }

    const selectedModels = getSelectedModels();
    console.log('Selected models:', selectedModels); // Debug log
    console.log('Recorded audio size:', recordedAudio ? recordedAudio.size : 'null'); // Debug log
    
    if (selectedModels.length === 0) {
        showError('Please select at least one transcription model');
        return;
    }

    // Show results section
    resultsSection.style.display = 'block';
    transcriptionResults.innerHTML = '';

    // Start transcription for each selected model
    const transcriptionPromises = selectedModels.map(model => 
        transcribeWithModel(model, recordedAudio)
    );

    // Wait for all transcriptions to complete
    const results = await Promise.allSettled(transcriptionPromises);
    
    // Save successful results to history
    results.forEach((result, index) => {
        if (result.status === 'fulfilled' && result.value.transcription) {
            saveToHistory({
                model: selectedModels[index],
                transcription: result.value.transcription,
                timestamp: new Date().toISOString(),
                duration: result.value.duration || 0
            });
        }
    });

    // Reset recording state
    updateRecordingUI('idle');
    recordedAudio = null;
    audioPreview.innerHTML = '<p>Ready for next recording</p>';
}

function getSelectedModels() {
    const checkboxes = document.querySelectorAll('input[name="transcription-model"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

async function transcribeWithModel(modelName, audioBlob) {
    const resultId = `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Create result container
    const resultElement = createResultElement(resultId, modelName, 'loading');
    transcriptionResults.appendChild(resultElement);

    try {
        let transcription;
        let duration;

        if (modelName.startsWith('whisper-') && !modelName.includes('api')) {
            // Local Whisper models
            const result = await transcribeWithLocalWhisper(modelName, audioBlob);
            transcription = result.transcription;
            duration = result.transcription_time;
        } else if (modelName === 'openai-whisper') {
            // OpenAI API
            const result = await transcribeWithOpenAI(audioBlob);
            transcription = result.transcription;
            duration = result.duration;
        } else if (modelName === 'gemini-speech') {
            // Gemini API
            const result = await transcribeWithGemini(audioBlob);
            transcription = result.transcription;
            duration = result.duration;
        } else {
            throw new Error(`Unsupported model: ${modelName}`);
        }

        // Update result with success
        updateResultElement(resultId, 'success', transcription, duration);
        
        return { transcription, duration, model: modelName };

    } catch (error) {
        console.error(`Transcription error for ${modelName}:`, error);
        updateResultElement(resultId, 'error', error.message);
        throw error;
    }
}

async function transcribeWithLocalWhisper(modelName, audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    formData.append('model', modelName);

    const response = await fetch('/api/audio/transcribe', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Transcription failed');
    }

    return await response.json();
}

async function transcribeWithOpenAI(audioBlob) {
    // This would require OpenAI API key and implementation
    const response = await fetch('/api/transcribe/openai', {
        method: 'POST',
        body: createFormData(audioBlob, 'openai-whisper')
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'OpenAI transcription failed');
    }

    return await response.json();
}

async function transcribeWithGemini(audioBlob) {
    const response = await fetch('/api/transcribe/gemini', {
        method: 'POST',
        body: createFormData(audioBlob, 'gemini-speech')
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Gemini transcription failed');
    }

    return await response.json();
}

function createFormData(audioBlob, model) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    formData.append('model', model);
    return formData;
}

// UI helper functions
function createResultElement(id, modelName, status) {
    const div = document.createElement('div');
    div.className = 'transcription-result';
    div.id = id;
    
    div.innerHTML = `
        <div class="result-header">
            <span class="result-model">${getModelDisplayName(modelName)}</span>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="result-status ${status}">${getStatusText(status)}</span>
                <span class="result-time">${new Date().toLocaleTimeString()}</span>
            </div>
        </div>
        <div class="result-text">
            ${status === 'loading' ? '<div class="loading-indicator"></div> Transcribing...' : ''}
        </div>
        <div class="result-actions" style="display: none;">
            <button class="btn-copy" onclick="copyToClipboard('${id}')">📋 Copy</button>
        </div>
    `;
    
    return div;
}

function updateResultElement(id, status, text, duration = null) {
    const element = document.getElementById(id);
    if (!element) return;

    const statusElement = element.querySelector('.result-status');
    const textElement = element.querySelector('.result-text');
    const actionsElement = element.querySelector('.result-actions');

    statusElement.className = `result-status ${status}`;
    statusElement.textContent = getStatusText(status);

    if (status === 'success') {
        textElement.textContent = text;
        textElement.dataset.text = text; // Store for copying
        actionsElement.style.display = 'flex';
        
        if (duration) {
            const durationSpan = document.createElement('span');
            durationSpan.className = 'result-time';
            durationSpan.textContent = `(${duration.toFixed(2)}s)`;
            statusElement.parentNode.appendChild(durationSpan);
        }
    } else if (status === 'error') {
        textElement.innerHTML = `<span style="color: #dc3545;">Error: ${text}</span>`;
    }
}

function getModelDisplayName(modelName) {
    const displayNames = {
        'whisper-base': 'Whisper Base (Local)',
        'whisper-small': 'Whisper Small (Local)',
        'whisper-medium': 'Whisper Medium (Local)',
        'whisper-large-v2': 'Whisper Large-v2 (Local)',
        'whisper-large-v3': 'Whisper Large-v3 (Local)',
        'openai-whisper': 'OpenAI Whisper API',
        'gemini-speech': 'Gemini Speech API'
    };
    return displayNames[modelName] || modelName;
}

function getStatusText(status) {
    switch (status) {
        case 'loading': return 'Processing...';
        case 'success': return 'Success';
        case 'error': return 'Error';
        default: return status;
    }
}

function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    const textElement = element.querySelector('.result-text');
    const text = textElement.dataset.text || textElement.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        const copyBtn = element.querySelector('.btn-copy');
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
        showError('Failed to copy to clipboard');
    });
}

// History functions
function saveToHistory(entry) {
    transcriptionHistory.unshift(entry);
    
    // Limit history to 50 entries
    if (transcriptionHistory.length > 50) {
        transcriptionHistory = transcriptionHistory.slice(0, 50);
    }
    
    // Save to localStorage
    localStorage.setItem('transcriptionHistory', JSON.stringify(transcriptionHistory));
    
    // Update display
    displayHistory();
}

function loadHistory() {
    const saved = localStorage.getItem('transcriptionHistory');
    if (saved) {
        try {
            transcriptionHistory = JSON.parse(saved);
            displayHistory();
        } catch (error) {
            console.error('Failed to load history:', error);
            transcriptionHistory = [];
        }
    }
}

function displayHistory(filter = '') {
    if (transcriptionHistory.length === 0) {
        historyList.innerHTML = '<p class="no-history">No transcription history yet</p>';
        return;
    }

    let filteredHistory = transcriptionHistory;
    if (filter) {
        filteredHistory = transcriptionHistory.filter(entry =>
            entry.transcription.toLowerCase().includes(filter.toLowerCase()) ||
            entry.model.toLowerCase().includes(filter.toLowerCase())
        );
    }

    if (filteredHistory.length === 0) {
        historyList.innerHTML = '<p class="no-history">No matching history found</p>';
        return;
    }

    historyList.innerHTML = filteredHistory.map((entry, index) => `
        <div class="history-item" onclick="copyHistoryItem(${transcriptionHistory.indexOf(entry)})">
            <div class="history-item-date">
                ${new Date(entry.timestamp).toLocaleString()} - ${getModelDisplayName(entry.model)}
            </div>
            <div class="history-item-preview">
                ${entry.transcription}
            </div>
        </div>
    `).join('');
}

function copyHistoryItem(index) {
    const entry = transcriptionHistory[index];
    if (entry) {
        navigator.clipboard.writeText(entry.transcription).then(() => {
            showSuccess('Copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy:', err);
            showError('Failed to copy to clipboard');
        });
    }
}

function filterHistory() {
    const filter = historySearch.value;
    displayHistory(filter);
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all transcription history?')) {
        transcriptionHistory = [];
        localStorage.removeItem('transcriptionHistory');
        displayHistory();
        showSuccess('History cleared');
    }
}

// Utility functions
function showError(message) {
    // Create or update error message
    let errorDiv = document.getElementById('error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        `;
        document.body.appendChild(errorDiv);
    }
    
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }, 5000);
}

function showSuccess(message) {
    // Create or update success message
    let successDiv = document.getElementById('success-message');
    if (!successDiv) {
        successDiv = document.createElement('div');
        successDiv.id = 'success-message';
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        `;
        document.body.appendChild(successDiv);
    }
    
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (successDiv) {
            successDiv.style.display = 'none';
        }
    }, 3000);
}