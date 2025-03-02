import os
import uuid
import numpy as np
import json
import base64  # Add base64 import
from flask import Flask, render_template, request, jsonify, session
from flask.json.provider import DefaultJSONProvider
from werkzeug.utils import secure_filename
from modules.prompt_generator import generate_paragraph
from modules.speech_analyzer import analyze_speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure charts directory exists
CHARTS_DIR = os.path.join('static', 'charts')
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)

# Custom JSON encoder for NumPy types
class NumpyJSONEncoder(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Use the custom JSON encoder
app.json_provider_class = NumpyJSONEncoder
app.json = app.json_provider_class(app)

@app.route('/')
def index():
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    topic = request.form.get('topic')
    # Add this to your route to debug
    print("Form data:", request.form)
    speaking_context = request.form.get('speaking-context', 'general') 
    print(f"Selected context: {speaking_context}")
    
    if not topic:
        return jsonify({'success': False, 'error': 'No topic provided'})
    
    try:
        paragraph = generate_paragraph(topic, speaking_context)
        return jsonify({'success': True, 'paragraph': paragraph})
    except Exception as e:
        print(f"Error in generate_prompt: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file uploaded'})
    
    audio_file = request.files['audio']
    speaking_context = request.form.get('speaking-context', 'general')  # Get the speaking context
    free_speech = request.form.get('free_speech', 'false')  # Check if this is free speech mode
    speech_topic = request.form.get('speech_topic', '')  # Get optional speech topic
    
    # Get paragraph if provided (not present in free speech mode)
    paragraph = request.form.get('paragraph', '')
    
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'})
    
    try:
        # Create user-specific directory
        user_id = session.get('user_id', 'default')
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Save the audio file
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(user_dir, filename)
        audio_file.save(file_path)
        
        print(f"Audio saved to {file_path}")
        print(f"Speaking context: {speaking_context}")
        
        if free_speech == 'true':
            print("Free speech mode detected")
            print(f"Speech topic: {speech_topic}")
            # For free speech, we don't have a reference paragraph
            # Create a placeholder or use an empty string
            if speech_topic:
                # Use speech topic as reference context
                mock_paragraph = f"Free speech on the topic: {speech_topic}"
            else:
                mock_paragraph = "Free speech practice"
            
            # Analyze the speech with context but without reference text comparison
            print("Starting free speech analysis...")
            analysis_results = analyze_speech(file_path, mock_paragraph, speaking_context)
            print("Free speech analysis completed")
        else:
            # Regular prompt-based analysis
            print(f"Reference text: {paragraph}")
            print("Starting speech analysis...")
            analysis_results = analyze_speech(file_path, paragraph, speaking_context)
            print("Speech analysis completed")
        
        # Save chart images to disk instead of session
        charts_dir = os.path.join('static', 'charts', user_id)
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # Save each chart as an image file
        image_paths = {}
        for chart_name, b64_data in analysis_results['charts_data'].items():
            if chart_name != 'scores':  # Only process image data
                img_path = os.path.join(charts_dir, f"{chart_name}.png")
                rel_path = os.path.join('charts', user_id, f"{chart_name}.png")
                
                # Convert base64 to image and save
                if b64_data:
                    with open(os.path.join('static', rel_path), 'wb') as f:
                        f.write(base64.b64decode(b64_data))
                    image_paths[chart_name] = rel_path
        
        # Create a lightweight serializable version
        serializable_results = {
            'scores': {k: float(v) for k, v in analysis_results['scores'].items()},
            'recommendations': analysis_results['recommendations'],
            'transcription': analysis_results['transcription'],
            'charts_data': {
                'scores': [float(s) for s in analysis_results['charts_data']['scores']],
                'image_paths': image_paths  # Store paths instead of images
            },
            'context': speaking_context,  # Store the context for the results page
            'is_free_speech': free_speech == 'true'  # Flag for results template
        }
        
        # Save results to session for the results page
        session['analysis_results'] = serializable_results
        
        # Save paragraph or speech topic for reference
        if free_speech == 'true':
            session['paragraph'] = f"Free speech on topic: {speech_topic}" if speech_topic else "Free speech practice"
            session['speech_topic'] = speech_topic
        else:
            session['paragraph'] = paragraph
            
        session['speaking_context'] = speaking_context  # Save context in session
        
        print("Results stored in session")
        return jsonify({
            'success': True, 
            'redirect': '/results'
        })
        
    except Exception as e:
        print(f"Error in upload_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/results')
def results():
    analysis_results = session.get('analysis_results')
    paragraph = session.get('paragraph')
    
    if not analysis_results:
        return render_template('index.html', error="No analysis results found. Please try again.")
    
    # Check if this was a free speech analysis
    is_free_speech = analysis_results.get('is_free_speech', False)
    speech_topic = session.get('speech_topic', '')
    
    return render_template(
        'results.html', 
        scores=analysis_results['scores'],
        recommendations=analysis_results['recommendations'],
        transcription=analysis_results['transcription'],
        paragraph=paragraph,
        charts_data=analysis_results['charts_data'],
        is_free_speech=is_free_speech,
        speech_topic=speech_topic
    )

if __name__ == '__main__':
    app.run(debug=True)