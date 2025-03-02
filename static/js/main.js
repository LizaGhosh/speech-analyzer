/**
 * Speech Practice Platform - Main JavaScript
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if browser supports audio recording
    if (!AudioRecorder.isRecordingSupported()) {
        alert('Your browser does not support audio recording. Please try using Chrome, Firefox, or Edge.');
        return;
    }
    
    // Initialize the recorder
    const recorder = new AudioRecorder();
    const freeRecorder = new AudioRecorder();
    
    // DOM Elements - Practice Methods
    const methodOptions = document.querySelectorAll('.method-option');
    const promptSection = document.getElementById('prompt-section');
    const freeSpeechSection = document.getElementById('free-speech-section');
    
    // DOM Elements - Prompt Generation
    const promptForm = document.getElementById('prompt-form');
    const topicInput = document.getElementById('topic');
    const speakingContextSelect = document.getElementById('speaking-context');
    const promptResult = document.getElementById('prompt-result');
    const paragraphDisplay = document.getElementById('paragraph-display');
    const recordingSection = document.getElementById('recording-section');
    
    // DOM Elements - Prompt Recording
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const recordingTimer = document.getElementById('recording-timer');
    const audioPreview = document.getElementById('audio-preview');
    const audioPlayback = document.getElementById('audio-playback');
    const retryBtn = document.getElementById('retry-btn');
    const submitBtn = document.getElementById('submit-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // DOM Elements - Free Speech
    const freeSpeakingContextSelect = document.getElementById('free-speaking-context');
    const speechTopicInput = document.getElementById('speech-topic');
    const freeRecordBtn = document.getElementById('free-record-btn');
    const freeStopBtn = document.getElementById('free-stop-btn');
    const freeRecordingTimer = document.getElementById('free-recording-timer');
    const freeAudioPreview = document.getElementById('free-audio-preview');
    const freeAudioPlayback = document.getElementById('free-audio-playback');
    const freeRetryBtn = document.getElementById('free-retry-btn');
    const freeSubmitBtn = document.getElementById('free-submit-btn');
    const freeLoadingIndicator = document.getElementById('free-loading-indicator');
    
    // Current paragraph for practice
    let currentParagraph = '';
    let currentContext = '';
    
    // Event Listeners - Practice Methods
    methodOptions.forEach(option => {
        option.addEventListener('click', function() {
            const method = this.getAttribute('data-method');
            
            // Update active class
            methodOptions.forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
            
            // Show/hide relevant sections
            if (method === 'generate-prompt') {
                promptSection.classList.remove('hidden');
                freeSpeechSection.classList.add('hidden');
                recordingSection.classList.add('hidden');
                promptResult.classList.add('hidden');
            } else if (method === 'free-speech') {
                promptSection.classList.add('hidden');
                freeSpeechSection.classList.remove('hidden');
                recordingSection.classList.add('hidden');
                promptResult.classList.add('hidden');
            }
        });
    });
    
    // Event Listeners - Prompt Generation
    promptForm.addEventListener('submit', generatePrompt);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    retryBtn.addEventListener('click', resetRecording);
    submitBtn.addEventListener('click', submitRecording);
    
    // Event Listeners - Free Speech
    freeRecordBtn.addEventListener('click', startFreeRecording);
    freeStopBtn.addEventListener('click', stopFreeRecording);
    freeRetryBtn.addEventListener('click', resetFreeRecording);
    freeSubmitBtn.addEventListener('click', submitFreeRecording);
    
    /**
     * Generate a prompt paragraph from the user's topic and context
     * @param {Event} e - The submit event
     */
    function generatePrompt(e) {
        e.preventDefault();
        
        const topic = topicInput.value.trim();
        if (!topic) {
            alert('Please enter a topic');
            return;
        }
        
        // Get the selected speaking context
        const speakingContext = speakingContextSelect.value;
        
        // Show loading state
        promptForm.querySelector('button').disabled = true;
        promptForm.querySelector('button').textContent = 'Generating...';
        
        // Send request to server
        const formData = new FormData();
        formData.append('topic', topic);
        formData.append('speaking-context', speakingContext);
        
        fetch('/generate_prompt', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset form state
            promptForm.querySelector('button').disabled = false;
            promptForm.querySelector('button').textContent = 'Generate Paragraph';
            
            if (data.success) {
                // Display generated paragraph
                currentParagraph = data.paragraph;
                currentContext = speakingContext;
                paragraphDisplay.textContent = currentParagraph;
                promptResult.classList.remove('hidden');
                recordingSection.classList.remove('hidden');
                
                // Scroll to recording section
                recordingSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error generating prompt: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            promptForm.querySelector('button').disabled = false;
            promptForm.querySelector('button').textContent = 'Generate Paragraph';
            alert('An error occurred. Please try again.');
        });
    }
    
    /**
     * Start recording audio (prompt method)
     */
    async function startRecording() {
        const success = await recorder.startRecording();
        
        if (success) {
            // Update UI
            recordBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
            recordingTimer.classList.remove('hidden');
            audioPreview.classList.add('hidden');
        } else {
            alert('Could not start recording. Please make sure your microphone is connected and you have granted permission to use it.');
        }
    }
    
    /**
     * Stop recording audio (prompt method)
     */
    async function stopRecording() {
        const audioBlob = await recorder.stopRecording();
        
        if (audioBlob) {
            // Update UI
            recordBtn.classList.remove('hidden');
            stopBtn.classList.add('hidden');
            audioPreview.classList.remove('hidden');
            
            // Set audio playback
            const recording = recorder.getRecording();
            audioPlayback.src = recording.url;
        }
    }
    
    /**
     * Reset recording state (prompt method)
     */
    function resetRecording() {
        audioPreview.classList.add('hidden');
        recordBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        recordingTimer.classList.add('hidden');
        recordingTimer.textContent = '00:00';
    }
    
    /**
     * Submit recording for analysis (prompt method)
     */
    function submitRecording() {
        if (!currentParagraph) {
            alert('Please generate a prompt paragraph first.');
            return;
        }
        
        const recording = recorder.getRecording();
        if (!recording.blob) {
            alert('Please record your speech first.');
            return;
        }
        
        // Show loading
        loadingIndicator.classList.remove('hidden');
        audioPreview.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('audio', recording.blob, 'recording.wav');
        formData.append('paragraph', currentParagraph);
        formData.append('speaking-context', currentContext);
        
        // Send to server
        fetch('/upload_audio', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.classList.add('hidden');
            
            if (data.success && data.redirect) {
                // Redirect to results page
                window.location.href = data.redirect;
            } else {
                alert('Error analyzing speech: ' + (data.error || 'Unknown error'));
                audioPreview.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.classList.add('hidden');
            audioPreview.classList.remove('hidden');
            alert('An error occurred while analyzing your speech. Please try again.');
        });
    }
    
    /**
     * Start recording audio (free speech method)
     */
    async function startFreeRecording() {
        const success = await freeRecorder.startRecording();
        
        if (success) {
            // Update UI
            freeRecordBtn.classList.add('hidden');
            freeStopBtn.classList.remove('hidden');
            freeRecordingTimer.classList.remove('hidden');
            freeAudioPreview.classList.add('hidden');
        } else {
            alert('Could not start recording. Please make sure your microphone is connected and you have granted permission to use it.');
        }
    }
    
    /**
     * Stop recording audio (free speech method)
     */
    async function stopFreeRecording() {
        const audioBlob = await freeRecorder.stopRecording();
        
        if (audioBlob) {
            // Update UI
            freeRecordBtn.classList.remove('hidden');
            freeStopBtn.classList.add('hidden');
            freeAudioPreview.classList.remove('hidden');
            
            // Set audio playback
            const recording = freeRecorder.getRecording();
            freeAudioPlayback.src = recording.url;
        }
    }
    
    /**
     * Reset recording state (free speech method)
     */
    function resetFreeRecording() {
        freeAudioPreview.classList.add('hidden');
        freeRecordBtn.classList.remove('hidden');
        freeStopBtn.classList.add('hidden');
        freeRecordingTimer.classList.add('hidden');
        freeRecordingTimer.textContent = '00:00';
    }
    
    /**
     * Submit free speech recording for analysis
     */
    function submitFreeRecording() {
        const speakingContext = freeSpeakingContextSelect.value;
        const speechTopic = speechTopicInput.value.trim();
        
        const recording = freeRecorder.getRecording();
        if (!recording.blob) {
            alert('Please record your speech first.');
            return;
        }
        
        // Show loading
        freeLoadingIndicator.classList.remove('hidden');
        freeAudioPreview.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('audio', recording.blob, 'recording.wav');
        formData.append('speaking-context', speakingContext);
        formData.append('free_speech', 'true');
        
        if (speechTopic) {
            formData.append('speech_topic', speechTopic);
        }
        
        // Send to server
        fetch('/upload_audio', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            freeLoadingIndicator.classList.add('hidden');
            
            if (data.success && data.redirect) {
                // Redirect to results page
                window.location.href = data.redirect;
            } else {
                alert('Error analyzing speech: ' + (data.error || 'Unknown error'));
                freeAudioPreview.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            freeLoadingIndicator.classList.add('hidden');
            freeAudioPreview.classList.remove('hidden');
            alert('An error occurred while analyzing your speech. Please try again.');
        });
    }
});