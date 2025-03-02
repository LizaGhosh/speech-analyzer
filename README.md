# Speech Analyzer Platform

A web-based platform that helps users improve their communication skills by analyzing their speaking performance.

## Features

- Generate practice paragraphs on any topic
- Record and analyze speech with detailed feedback
- Analyze voice modulation, pauses, expression, clarity, and speaking pace
- Context-specific feedback based on different speaking scenarios (interviews, debates, presentations, etc.)
- Visual analytics with waveform, pitch contour, and volume analysis

## Technical Stack

- Python/Flask backend
- Web Audio API for in-browser recording
- Wav2Vec2 for speech-to-text and analysis
- Librosa for audio feature extraction
- Chart.js for visualization

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file: GEMINI_API_KEY=your_api_key_here
4. Run the app: `python app.py`
5. Access at http://localhost:5000

## Usage

1. Choose between generated prompt practice or free speech
2. For generated prompts: Enter a topic and speaking context
3. Record your speech by reading the generated paragraph
4. Receive detailed analysis and personalized feedback