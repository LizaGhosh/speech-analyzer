/**
 * Audio Recorder Class
 * Handles recording audio using the MediaRecorder API
 */
class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.audioUrl = null;
        this.stream = null;
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        this.timerElement = null;
        
        // Bind methods
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.updateTimer = this.updateTimer.bind(this);
        this.getRecording = this.getRecording.bind(this);
    }
    
    /**
     * Start recording audio
     * @param {Function} onDataAvailable - Callback when data is available (optional)
     * @param {HTMLElement} timerElement - Element to update with timer (optional)
     * @returns {Promise} - Resolves when recording starts
     */
    async startRecording(onDataAvailable = null, timerElement = null) {
        try {
            // Reset recording state
            this.audioChunks = [];
            this.audioBlob = null;
            this.audioUrl = null;
            this.timerElement = timerElement || document.getElementById('recording-timer');
            
            // Get user media
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.stream);
            
            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    if (onDataAvailable) {
                        onDataAvailable(event.data);
                    }
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                this.isRecording = false;
                
                // Stop all tracks in the stream
                this.stream.getTracks().forEach(track => track.stop());
                
                // Stop timer
                clearInterval(this.timerInterval);
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            
            // Start timer
            this.startTime = Date.now();
            this.updateTimer();
            this.timerInterval = setInterval(this.updateTimer, 1000);
            
            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            return false;
        }
    }
    
    /**
     * Stop recording audio
     * @returns {Promise} - Resolves with the audio blob when recording stops
     */
    stopRecording() {
        return new Promise((resolve) => {
            if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
                resolve(null);
                return;
            }
            
            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                this.isRecording = false;
                
                // Stop all tracks in the stream
                this.stream.getTracks().forEach(track => track.stop());
                
                // Stop timer
                clearInterval(this.timerInterval);
                
                resolve(this.audioBlob);
            };
            
            this.mediaRecorder.stop();
        });
    }
    
    /**
     * Update recording timer display
     */
    updateTimer() {
        if (!this.startTime || !this.timerElement) return;
        
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        
        this.timerElement.textContent = `${minutes}:${seconds}`;
    }
    
    /**
     * Get the recorded audio
     * @returns {Object} - The audio blob and URL
     */
    getRecording() {
        return {
            blob: this.audioBlob,
            url: this.audioUrl
        };
    }
    
    /**
     * Check if browser supports recording
     * @returns {Boolean} - True if recording is supported
     */
    static isRecordingSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }
}