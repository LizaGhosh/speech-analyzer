import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import json
import time
import base64
from io import BytesIO
from difflib import SequenceMatcher
import re

class SpeechAnalyzer:
    """Enhanced speech analysis system that uses Wav2Vec2.0 with context-aware feedback."""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """Initialize the analyzer with the specified model."""
        # Load Wav2Vec2.0 components
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Set device (use CUDA if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Initialize variables
        self.audio_data = None
        self.sample_rate = None
        self.features = {}
        self.scores = {}
        self.transcription = None
        self.reference_text = None
        self.context = "general"  # Default context
        
        # Common filler words for detection
        self.filler_words = ["um", "uh", "er", "ah", "like", "you know", "so", "actually", "basically"]
        
        # Context-specific ideal ranges for speech parameters
        self.context_parameters = {
            "general": {
                "optimal_pause_ratio": 0.25,
                "min_pauses_per_minute": 8,
                "max_pauses_per_minute": 20,
                "min_wpm": 130,
                "max_wpm": 170,
                "optimal_pitch_variation": 40
            },
            "interview": {
                "optimal_pause_ratio": 0.25,
                "min_pauses_per_minute": 8,
                "max_pauses_per_minute": 18,
                "min_wpm": 120,
                "max_wpm": 150,
                "optimal_pitch_variation": 35
            },
            "debate": {
                "optimal_pause_ratio": 0.20,
                "min_pauses_per_minute": 6,
                "max_pauses_per_minute": 15,
                "min_wpm": 140,
                "max_wpm": 180,
                "optimal_pitch_variation": 45
            },
            "storytelling": {
                "optimal_pause_ratio": 0.30,
                "min_pauses_per_minute": 10,
                "max_pauses_per_minute": 22,
                "min_wpm": 120,
                "max_wpm": 160,
                "optimal_pitch_variation": 50
            },
            "business_presentation": {
                "optimal_pause_ratio": 0.28,
                "min_pauses_per_minute": 9,
                "max_pauses_per_minute": 18,
                "min_wpm": 130,
                "max_wpm": 160,
                "optimal_pitch_variation": 40
            },
            "casual": {
                "optimal_pause_ratio": 0.22,
                "min_pauses_per_minute": 7,
                "max_pauses_per_minute": 25,
                "min_wpm": 140,
                "max_wpm": 180,
                "optimal_pitch_variation": 30
            },
            "teaching": {
                "optimal_pause_ratio": 0.30,
                "min_pauses_per_minute": 10,
                "max_pauses_per_minute": 20,
                "min_wpm": 120,
                "max_wpm": 150,
                "optimal_pitch_variation": 35
            }
        }
        
    def load_audio(self, file_path):
        """Load and preprocess the audio file."""
        # Convert to wav if needed
        if not file_path.endswith('.wav'):
            wav_path = file_path.replace(os.path.splitext(file_path)[1], '.wav')
            sound = AudioSegment.from_file(file_path)
            sound.export(wav_path, format="wav")
            file_path = wav_path
        
        # Load the audio file
        self.audio_data, self.sample_rate = librosa.load(file_path, sr=16000)  # Wav2Vec2 expects 16kHz
        return self
    
    def transcribe_audio(self):
        """Transcribe the audio using Wav2Vec2.0."""
        if self.audio_data is None:
            raise ValueError("No audio loaded. Call load_audio() first.")
        
        inputs = self.processor(self.audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert ids to text
        self.transcription = self.processor.batch_decode(predicted_ids)[0]
        return self.transcription
    
    def set_reference_text(self, text):
        """Set the reference text to compare with transcription."""
        self.reference_text = text
        return self
    
    def set_context(self, context):
        """Set the speaking context for analysis."""
        if context in self.context_parameters:
            self.context = context
        else:
            print(f"Warning: Unknown context '{context}'. Using 'general' instead.")
            self.context = "general"
        return self
    
    def extract_features(self):
        """Extract enhanced speech features for analysis."""
        if self.audio_data is None:
            raise ValueError("No audio loaded. Call load_audio() first.")
        
        # Basic features
        self.features['duration'] = len(self.audio_data) / self.sample_rate
        
        # === Voice Activity Detection & Pause Analysis ===
        # Calculate energy
        energy = librosa.feature.rms(y=self.audio_data, frame_length=1024, hop_length=512)[0]
        energy_threshold = np.mean(energy) * 0.5
        
        # Find active and silent frames
        active_frames = energy > energy_threshold
        silent_frames = ~active_frames
        
        # Convert frames to time
        frame_time = librosa.frames_to_time(np.arange(len(energy)), sr=self.sample_rate, hop_length=512)
        
        # Identify silent segments and their timestamps
        silent_segments = []
        current_segment = None
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and current_segment is None:
                current_segment = [frame_time[i]]
            elif not is_silent and current_segment is not None:
                current_segment.append(frame_time[i-1])
                if current_segment[1] - current_segment[0] >= 0.3:  # Only count pauses > 300ms
                    silent_segments.append(current_segment)
                current_segment = None
        
        # Add last segment if exists
        if current_segment is not None:
            current_segment.append(frame_time[-1])
            if current_segment[1] - current_segment[0] >= 0.3:
                silent_segments.append(current_segment)
        
        # Store pause timestamps for feedback
        self.features['pause_timestamps'] = silent_segments
        
        # Calculate pause features
        self.features['pause_count'] = len(silent_segments)
        
        if silent_segments:
            pause_durations = [end - start for start, end in silent_segments]
            self.features['mean_pause_duration'] = np.mean(pause_durations)
            self.features['total_pause_duration'] = np.sum(pause_durations)
            self.features['pause_ratio'] = self.features['total_pause_duration'] / self.features['duration']
            self.features['pauses_per_minute'] = self.features['pause_count'] / (self.features['duration'] / 60)
            
            # Calculate pause distribution (short, medium, long pauses)
            short_pauses = [d for d in pause_durations if 0.3 <= d < 1.0]
            medium_pauses = [d for d in pause_durations if 1.0 <= d < 2.0]
            long_pauses = [d for d in pause_durations if d >= 2.0]
            
            self.features['short_pauses'] = len(short_pauses)
            self.features['medium_pauses'] = len(medium_pauses)
            self.features['long_pauses'] = len(long_pauses)
        else:
            self.features['mean_pause_duration'] = 0
            self.features['total_pause_duration'] = 0
            self.features['pause_ratio'] = 0
            self.features['pauses_per_minute'] = 0
            self.features['short_pauses'] = 0
            self.features['medium_pauses'] = 0
            self.features['long_pauses'] = 0
        
        # === Pitch Analysis (for modulation) ===
        # Using librosa's pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio_data, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Store pitch data for visualization and feedback
        self.features['f0'] = f0
        self.features['voiced_flag'] = voiced_flag
        self.features['pitch_times'] = librosa.times_like(f0, sr=self.sample_rate)
        
        # Filter to get only voiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            self.features['pitch_mean'] = np.mean(f0_voiced)
            self.features['pitch_std'] = np.std(f0_voiced)
            self.features['pitch_range'] = np.max(f0_voiced) - np.min(f0_voiced)
            
            # Calculate pitch dynamics (changes in pitch)
            pitch_changes = np.diff(f0_voiced)
            self.features['pitch_dynamics_mean'] = np.mean(np.abs(pitch_changes))
            
            # Compute pitch variability over time (for visualization)
            # Split the audio into segments and analyze pitch variation in each
            segment_length = int(self.sample_rate * 2)  # 2-second segments
            num_segments = int(len(self.audio_data) / segment_length)
            
            segment_pitch_variations = []
            segment_times = []
            
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(self.audio_data))
                segment_time = start_idx / self.sample_rate
                
                # Find pitch values in this segment
                start_time = segment_time
                end_time = end_idx / self.sample_rate
                
                pitch_indices = np.where((self.features['pitch_times'] >= start_time) & 
                                        (self.features['pitch_times'] < end_time))[0]
                
                if len(pitch_indices) > 0:
                    segment_pitches = f0[pitch_indices]
                    segment_voiced = voiced_flag[pitch_indices]
                    if np.any(segment_voiced):
                        segment_f0_voiced = segment_pitches[segment_voiced]
                        if len(segment_f0_voiced) > 1:
                            variation = np.std(segment_f0_voiced)
                            segment_pitch_variations.append(variation)
                            segment_times.append(segment_time)
            
            self.features['segment_pitch_variations'] = segment_pitch_variations
            self.features['segment_times'] = segment_times
        else:
            self.features['pitch_mean'] = 0
            self.features['pitch_std'] = 0
            self.features['pitch_range'] = 0
            self.features['pitch_dynamics_mean'] = 0
            self.features['segment_pitch_variations'] = []
            self.features['segment_times'] = []
        
        # === Volume/Intensity Analysis ===
        # RMS energy for overall volume
        rms = librosa.feature.rms(y=self.audio_data)[0]
        self.features['volume_mean'] = np.mean(rms)
        self.features['volume_std'] = np.std(rms)
        self.features['volume_range'] = np.max(rms) - np.min(rms)
        
        # Store volume data for visualization
        self.features['volume_data'] = rms
        self.features['volume_times'] = librosa.times_like(rms, sr=self.sample_rate, hop_length=512)
        
        # Volume dynamics (changes in volume)
        volume_changes = np.diff(rms)
        self.features['volume_dynamics_mean'] = np.mean(np.abs(volume_changes))
        
        # Find segments with low volume (potential mumbling)
        low_volume_threshold = np.mean(rms) * 0.7
        low_volume_frames = rms < low_volume_threshold
        
        # Convert frames to time
        low_volume_times = librosa.frames_to_time(np.where(low_volume_frames)[0], 
                                                sr=self.sample_rate, 
                                                hop_length=512)
        
        self.features['low_volume_times'] = low_volume_times
        
        # === Spectral features (for clarity and voice quality) ===
        # Spectral centroid (brightness of sound)
        spec_cent = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate)[0]
        self.features['spectral_centroid_mean'] = np.mean(spec_cent)
        self.features['spectral_centroid_std'] = np.std(spec_cent)
        
        # === Speech rate analysis ===
        if self.transcription:
            # Count words
            word_count = len(self.transcription.split())
            self.features['word_count'] = word_count
            self.features['speaking_rate'] = word_count / (self.features['duration'] / 60)  # words per minute
            
            # Detect filler words
            filler_count = 0
            filler_matches = []
            
            for filler in self.filler_words:
                # Find all occurrences of the filler word
                for match in re.finditer(rf'\b{re.escape(filler)}\b', self.transcription.lower()):
                    filler_count += 1
                    filler_matches.append((match.group(), match.start()))
            
            self.features['filler_count'] = filler_count
            self.features['filler_matches'] = filler_matches
            self.features['filler_frequency'] = filler_count / word_count if word_count > 0 else 0
        else:
            self.features['word_count'] = 0
            self.features['speaking_rate'] = 0
            self.features['filler_count'] = 0
            self.features['filler_matches'] = []
            self.features['filler_frequency'] = 0
        
        return self
    
    def calculate_accuracy(self):
        """Calculate accuracy by comparing transcription to reference text."""
        if not self.transcription or not self.reference_text:
            return 0
        
        # Clean texts for comparison
        clean_trans = self.transcription.lower()
        clean_ref = self.reference_text.lower()
        
        # Use sequence matcher to get similarity ratio
        matcher = SequenceMatcher(None, clean_trans, clean_ref)
        similarity = matcher.ratio()
        
        return similarity * 100  # Convert to percentage
    
    def calculate_scores(self, context="general"):
        """Calculate scores for voice modulation, pauses, and expressions based on context."""
        if not self.features:
            self.extract_features()
            
        # Set the context for analysis
        self.set_context(context)
        
        # Get context-specific parameters
        params = self.context_parameters[self.context]
        
        # Define scoring weight adjustments per context
        scoring_weights = {
            "general": {"modulation": 0.3, "pauses": 0.2, "expression": 0.2, "accuracy": 0.3},
            "interview": {"modulation": 0.25, "pauses": 0.3, "expression": 0.15, "accuracy": 0.3},
            "debate": {"modulation": 0.35, "pauses": 0.2, "expression": 0.25, "accuracy": 0.2},
            "storytelling": {"modulation": 0.35, "pauses": 0.2, "expression": 0.35, "accuracy": 0.1},
            "business_presentation": {"modulation": 0.25, "pauses": 0.3, "expression": 0.15, "accuracy": 0.3},
            "casual": {"modulation": 0.2, "pauses": 0.2, "expression": 0.3, "accuracy": 0.3},
            "teaching": {"modulation": 0.2, "pauses": 0.3, "expression": 0.2, "accuracy": 0.3}
        }
        
        # Use the weights for the current context
        weights = scoring_weights.get(self.context, scoring_weights["general"])
        
        # 1. Voice Modulation Score (0-100)
        # Normalize features for scoring using context-specific optimal values
        pitch_std_normalized = min(1.0, self.features['pitch_std'] / params['optimal_pitch_variation'])
        pitch_range_normalized = min(1.0, self.features['pitch_range'] / 200.0)  # Normalize pitch range
        pitch_dynamics_normalized = min(1.0, self.features['pitch_dynamics_mean'] / 20.0)  # Normalize pitch changes
        
        # Combine normalized features with weights
        modulation_score = (
            0.4 * pitch_std_normalized + 
            0.3 * pitch_range_normalized + 
            0.3 * pitch_dynamics_normalized
        ) * 100
        
        # 2. Pauses Score (0-100)
        # Use context-specific optimal pause ratio
        optimal_pause_ratio = params['optimal_pause_ratio']
        pause_ratio_score = max(0, 1 - abs(self.features['pause_ratio'] - optimal_pause_ratio) * 5)
        
        # Use context-specific ideal pauses per minute
        if self.features['pauses_per_minute'] < params['min_pauses_per_minute']:
            pauses_freq_score = self.features['pauses_per_minute'] / params['min_pauses_per_minute']
        elif self.features['pauses_per_minute'] > params['max_pauses_per_minute']:
            pauses_freq_score = max(0, 1 - (self.features['pauses_per_minute'] - params['max_pauses_per_minute']) / 10)
        else:
            pauses_freq_score = 1.0
            
        # Combine pause metrics with weights
        pause_score = (0.5 * pause_ratio_score + 0.5 * pauses_freq_score) * 100
        
        # 3. Expression Score (0-100)
        # Combine volume dynamics and spectral variations
        volume_dynamics_normalized = min(1.0, self.features['volume_dynamics_mean'] / 0.05)
        spectral_variation_normalized = min(1.0, self.features['spectral_centroid_std'] / 500)
        
        # Penalize for excessive filler words
        filler_penalty = max(0, 1 - (self.features['filler_frequency'] * 5))  # Reduce score if fillers > 20% of words
        
        expression_score = (
            0.5 * volume_dynamics_normalized + 
            0.3 * spectral_variation_normalized +
            0.2 * filler_penalty
        ) * 100
        
        # 4. Accuracy Score
        accuracy_score = self.calculate_accuracy()
        
        # 5. Pace Score (based on speaking rate)
        if self.features['speaking_rate'] > 0:
            if self.features['speaking_rate'] < params['min_wpm']:
                pace_score = (self.features['speaking_rate'] / params['min_wpm']) * 100
            elif self.features['speaking_rate'] > params['max_wpm']:
                pace_score = max(0, (1 - (self.features['speaking_rate'] - params['max_wpm']) / 50)) * 100
            else:
                pace_score = 100
        else:
            pace_score = 0
        
        # Calculate overall score with context-specific weights
        overall_score = (
            weights["modulation"] * modulation_score + 
            weights["pauses"] * pause_score + 
            weights["expression"] * expression_score + 
            weights["accuracy"] * accuracy_score
        )
        
        # Store scores
        self.scores = {
            'modulation': round(modulation_score, 1),
            'pauses': round(pause_score, 1),
            'expression': round(expression_score, 1),
            'accuracy': round(accuracy_score, 1),
            'pace': round(pace_score, 1),
            'overall': round(overall_score, 1)
        }
        
        return self.scores
    
    def generate_recommendations(self, context="general"):
        """Generate feedback and recommendations based on scores and speaking context."""
        # Ensure we have the right context and scores are calculated
        self.set_context(context)
        if not self.scores:
            self.calculate_scores(context)
            
        recommendations = []
        
        # FORMAT TIMESTAMPS
        def format_time(seconds):
            """Format seconds into MM:SS format."""
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
            
        # CONTEXT-SPECIFIC RECOMMENDATIONS
        # --------------------------------
        
        # MODULATION RECOMMENDATIONS
        if context == "interview":
            # Modulation recommendations for interviews
            if self.scores['modulation'] < 60:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your tone lacks the engagement needed for an interview. When discussing your qualifications or achievements, try raising your pitch slightly to convey enthusiasm. Record yourself answering key interview questions and practice varying your tone on important words like 'led', 'achieved', or 'succeeded'."
                })
            elif self.scores['modulation'] < 80:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your pitch variation is adequate but could be more engaging for an interview. When describing past achievements, emphasize the impact with slightly higher pitch to show enthusiasm. Practice by recording answers to questions about your strengths and accomplishments."
                })
        
        elif context == "debate":
            # Modulation recommendations for debates
            if self.scores['modulation'] < 70:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your voice lacks the dynamic variation needed for persuasive debate. Try using a wider pitch range to emphasize key arguments. Study effective debaters who use vocal emphasis to drive points home. Practice by recording yourself making the same point with different emphasis patterns."
                })
            elif self.scores['modulation'] < 85:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your pitch variation is good but could be more strategic for debate. Develop a pattern of building vocal intensity through your arguments, culminating in key points with stronger emphasis. This creates a persuasive rhythm that audiences find compelling."
                })
                
        elif context == "storytelling":
            # Modulation recommendations for storytelling
            if self.scores['modulation'] < 70:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your voice lacks the expressiveness that makes storytelling engaging. Try using different voices or tones for dialogue, and vary your pitch to build suspense or express emotions. Practice by recording yourself reading children's stories, where exaggerated expression is appropriate."
                })
            elif self.scores['modulation'] < 85:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your voice modulation is good but could be more dynamic for storytelling. Create more contrast between narrative sections and dramatic moments by increasing pitch variation during emotional peaks. Listen to audiobook narrators or podcast storytellers for inspiration."
                })
                
        elif context == "business_presentation":
            # Modulation recommendations for business presentations
            if self.scores['modulation'] < 65:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your delivery is too flat for an effective business presentation. Even in professional settings, vocal variety maintains audience engagement. Try emphasizing key metrics, conclusions, or recommendations with slightly higher pitch. Record yourself presenting data, then again with deliberate emphasis on the implications."
                })
            elif self.scores['modulation'] < 80:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your pitch variation is adequate but could be more strategic for business presentations. Use a deliberate pattern of neutral delivery for background information, then slightly elevated pitch for insights and key takeaways. This signals to your audience what they should remember."
                })
                
        elif context == "casual":
            # Modulation recommendations for casual conversation
            if self.scores['modulation'] < 60:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your tone sounds monotonous for casual conversation. Natural speech has organic pitch variations that convey interest and emotion. Practice by recording yourself telling a friend about something exciting that happened to you, focusing on letting your natural enthusiasm come through."
                })
            elif self.scores['modulation'] > 90:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your vocal variation might seem excessive for casual conversation. While expressiveness is good, extremely dramatic delivery can seem unnatural in everyday settings. Try moderating your pitch variations slightly for a more conversational tone."
                })
                
        elif context == "teaching":
            # Modulation recommendations for teaching
            if self.scores['modulation'] < 65:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your tone lacks the variation that keeps students engaged. Effective teachers use pitch changes to highlight important concepts and maintain attention. Try elevating your tone when introducing key points, and dropping it slightly when providing deeper explanations. Record yourself explaining a concept, then practice the same explanation with deliberate vocal emphasis."
                })
            elif self.scores['modulation'] < 80:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your pitch variation is good but could be more pedagogically effective. Develop a pattern of using higher pitch for questions and key concepts, with a return to baseline for elaboration. This rhythmic pattern helps students identify and remember important information."
                })
        
        else:  # General context
            # General modulation recommendations
            if self.scores['modulation'] < 60:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "Your voice has limited pitch variation. Try to vary your tone more when speaking, especially on important words. Practice reading with exaggerated pitch changes to develop this skill."
                })
            elif self.scores['modulation'] < 80:
                recommendations.append({
                    'category': 'modulation',
                    'type': 'improvement',
                    'text': "You have moderate voice modulation. To improve, consciously raise and lower your pitch for emphasis. Try to make your voice slightly higher when expressing important points or showing enthusiasm."
                })
                
        # PAUSE RECOMMENDATIONS
        # ---------------------
        
        # Check for pause-related issues specific to context
        if context == "interview":
            if self.scores['pauses'] < 60:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Your responses lack effective pauses for an interview setting. Without strategic pauses, your answers may seem rushed or rehearsed. Practice inserting a 1-2 second pause after making an important point about your experience or skills. This gives the interviewer time to absorb your qualifications."
                })
            elif self.features['pauses_per_minute'] > self.context_parameters[context]['max_pauses_per_minute']:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': f"You're using too many pauses ({round(self.features['pauses_per_minute'], 1)} per minute), which can make you seem hesitant in an interview. While some pauses show thoughtfulness, excessive pausing might suggest uncertainty. Practice giving more fluid responses to common interview questions."
                })
                
        elif context == "debate":
            if self.scores['pauses'] < 60:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Effective debaters use strategic pauses to emphasize key arguments and allow points to land with the audience. Your delivery lacks these rhetorical pauses. Try inserting a deliberate 1-second pause after stating your strongest points or before countering opposing arguments."
                })
            elif self.features['long_pauses'] > 3:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "You have several extended pauses that may undermine your debating momentum. While short pauses are effective for emphasis, long hesitations can suggest uncertainty. Practice transitioning more smoothly between arguments to maintain persuasive flow."
                })
                
        elif context == "storytelling":
            if self.scores['pauses'] < 65:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Your storytelling lacks the dramatic pauses that create suspense and engagement. Effective storytellers use silence strategically before revelations or dramatic moments. Practice pausing for 2-3 seconds before revealing surprising elements or key turning points in your narrative."
                })
            elif self.features['pause_ratio'] < 0.15:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Your storytelling feels rushed with insufficient pausing. Listeners need moments to visualize and process narrative elements. Try extending pauses after descriptive sections to let the imagery sink in, and before dramatic moments to build anticipation."
                })
                
        elif context == "business_presentation":
            if self.scores['pauses'] < 65:
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Your presentation lacks sufficient pauses for information retention. In business contexts, audiences need time to process data and insights. Practice pausing for 2-3 seconds after presenting key metrics or recommendations. Check your slides for transition moments that would benefit from a deliberate pause."
                })
            elif self.features['short_pauses'] > self.features['medium_pauses'] * 3:
                # Instead of using 'pause_distribution', check if there are many more short pauses than medium ones
                recommendations.append({
                    'category': 'pauses',
                    'type': 'improvement',
                    'text': "Your pauses are too brief for a business presentation. While you pause frequently, these short pauses don't give the audience enough time to process information. Try extending your pauses to 1-2 seconds after key points for better audience comprehension."
                })
        
        # Check for filler words (applies to all contexts)
        if self.features['filler_count'] > 5 and self.features['filler_frequency'] > 0.05:
            recommendations.append({
                'category': 'expression',
                'type': 'improvement',
                'text': f"You used filler words like 'um', 'uh', or 'like' approximately {self.features['filler_count']} times ({round(self.features['filler_frequency']*100, 1)}% of your words). Replace these with deliberate pauses instead. Practice by recording yourself and consciously pausing rather than using fillers when gathering your thoughts."
            })
            
        # PACE RECOMMENDATIONS
        # -------------------
        params = self.context_parameters[self.context]
        
        if self.features['speaking_rate'] > 0:  # Only if we have valid speaking rate data
            if self.features['speaking_rate'] < params['min_wpm']:
                recommendations.append({
                    'category': 'pace',
                    'type': 'improvement',
                    'text': f"Your speaking rate of {round(self.features['speaking_rate'])} words per minute is slower than ideal for {context.replace('_', ' ')} (recommended {params['min_wpm']}-{params['max_wpm']} WPM). This can reduce engagement and impact. Try practicing with a timer, aiming to cover slightly more content in the same time while maintaining clarity."
                })
            elif self.features['speaking_rate'] > params['max_wpm']:
                recommendations.append({
                    'category': 'pace',
                    'type': 'improvement',
                    'text': f"Your speaking rate of {round(self.features['speaking_rate'])} words per minute is faster than ideal for {context.replace('_', ' ')} (recommended {params['min_wpm']}-{params['max_wpm']} WPM). This may reduce clarity and comprehension. Practice slowing down by slightly extending vowel sounds and inserting more deliberate pauses between sentences."
                })
            elif context == "debate" and self.features['speaking_rate'] < params['max_wpm'] - 10:
                recommendations.append({
                    'category': 'pace',
                    'type': 'improvement',
                    'text': f"For debate, your pace of {round(self.features['speaking_rate'])} WPM could be more dynamic. Consider varying your pace - speaking more rapidly when summarizing established points, and slowing down when introducing new or complex arguments. This contrast in pace helps emphasize key points."
                })
        
        # EXPRESSION RECOMMENDATIONS
        # -------------------------
        if self.scores['expression'] < 60:
            if context == "storytelling":
                recommendations.append({
                    'category': 'expression',
                    'type': 'improvement',
                    'text': "Your storytelling lacks the emotional expressiveness that creates engagement. Practice varying your volume and tone to match the emotional content - softer for intimate or tense moments, louder for exciting or climactic ones. Record yourself reading the same passage with different emotional intentions to develop range."
                })
            elif context == "business_presentation":
                recommendations.append({
                    'category': 'expression',
                    'type': 'improvement',
                    'text': "Your presentation delivery lacks the expressiveness that maintains audience engagement. Even in professional settings, varying your delivery when transitioning from data to insights keeps listeners attentive. Practice emphasizing the 'so what' implications of your data points with slightly more animated delivery."
                })
            elif context == "debate":
                recommendations.append({
                    'category': 'expression',
                    'type': 'improvement',
                    'text': "Your arguments lack persuasive expressiveness. Effective debaters use vocal intensity to convey conviction. Practice building vocal energy through your key points, with appropriate emphasis on conclusive statements. Study political speakers who effectively use volume and vocal texture to persuade."
                })
            else:
                recommendations.append({
                    'category': 'expression',
                    'type': 'improvement',
                    'text': "Your delivery could benefit from more expressive energy. Practice adding emotional color to important words by varying volume and emphasis. Try recording yourself reading passages with different emotions to develop range."
                })
        
        # ACCURACY RECOMMENDATIONS
        # -----------------------
        if self.scores['accuracy'] < 60:
            recommendations.append({
                'category': 'accuracy',
                'type': 'improvement',
                'text': "Your speech differs significantly from the intended content. Focus on pronunciation clarity by speaking more slowly and articulating each word fully. Pay special attention to ending consonants which are often dropped in casual speech."
            })
        elif self.scores['accuracy'] < 80:
            recommendations.append({
                'category': 'accuracy',
                'type': 'improvement',
                'text': "Your pronunciation is generally clear, but some words weren't captured accurately. Practice articulating challenging words and maintain consistent volume throughout your speech."
            })
            
        # Add POSITIVE recommendations for high scores
        strengths = []
        
        if self.scores['modulation'] >= 80:
            strengths.append({
                'category': 'modulation',
                'type': 'positive',
                'text': f"Excellent voice modulation for {context.replace('_', ' ')}! You effectively use pitch variation to make your speech engaging and dynamic. Your tonal range helps emphasize important points appropriately."
            })
            
        if self.scores['pauses'] >= 80:
            strengths.append({
                'category': 'pauses',
                'type': 'positive',
                'text': f"Great use of pauses for {context.replace('_', ' ')}! You effectively use strategic pauses to structure your speech and emphasize important points. This gives listeners time to process your message."
            })
            
        if self.scores['expression'] >= 80:
            strengths.append({
                'category': 'expression',
                'type': 'positive',
                'text': f"Excellent expressiveness for {context.replace('_', ' ')}! Your voice conveys emotion and emphasis effectively, making your speech engaging and impactful. Your vocal variety maintains listener interest."
            })
            
        if self.scores['accuracy'] >= 85:
            strengths.append({
                'category': 'accuracy',
                'type': 'positive',
                'text': "Excellent pronunciation and clarity! Your words were easily recognized and understood, demonstrating strong articulation skills."
            })
            
        if self.scores['pace'] >= 85:
            strengths.append({
                'category': 'pace',
                'type': 'positive',
                'text': f"Your speaking pace of {round(self.features['speaking_rate'])} words per minute is ideal for {context.replace('_', ' ')}. This balanced rate allows for clear articulation while maintaining engagement."
            })
            
        # Add strengths to recommendations
        for strength in strengths:
            recommendations.append(strength)
        
        # Add general positive recommendation if we have multiple strengths
        if len(strengths) >= 2:
            top_strengths = [s['category'] for s in strengths[:2]]
            strength_text = f"{top_strengths[0]} and {top_strengths[1]}"
            
            recommendations.append({
                'category': 'general',
                'type': 'positive',
                'text': f"Your greatest strengths are your {strength_text}. Continue to build on these while working on other aspects of your delivery."
            })
        
        # Add SPECIFIC IMPROVEMENT EXERCISE based on weakest area
        scores_dict = {
            'modulation': self.scores['modulation'],
            'pauses': self.scores['pauses'],
            'expression': self.scores['expression'],
            'pace': self.scores['pace'],
            'accuracy': self.scores['accuracy']
        }
        
        weakest_area = min(scores_dict, key=scores_dict.get)
        
        if weakest_area == 'modulation':
            recommendations.append({
                'category': 'practice',
                'type': 'improvement',
                'text': "FOCUSED PRACTICE EXERCISE: Record yourself reading the same passage three different ways - first with minimal expression, then with moderate expression, and finally with exaggerated pitch variation. Compare the recordings to develop awareness of how different levels of modulation impact your delivery."
            })
        elif weakest_area == 'pauses':
            recommendations.append({
                'category': 'practice',
                'type': 'improvement',
                'text': "FOCUSED PRACTICE EXERCISE: Take a short paragraph and mark places for pauses with slashes - after commas (short pause), periods (medium pause), and before important points (longer pause). Practice reading with these deliberate pauses, timing them with a stopwatch for consistency."
            })
        elif weakest_area == 'expression':
            recommendations.append({
                'category': 'practice',
                'type': 'improvement',
                'text': "FOCUSED PRACTICE EXERCISE: Choose a paragraph with varied emotional content. Read it while deliberately changing your vocal expression to match each emotional shift. Record yourself and listen for authentic emotional connection. Practice until the transitions feel natural."
            })
        elif weakest_area == 'pace':
            recommendations.append({
                'category': 'practice',
                'type': 'improvement',
                'text': "FOCUSED PRACTICE EXERCISE: Find a 100-word passage and time yourself reading it. If you're speaking too quickly, aim to extend your time by 15-20% by slightly elongating vowel sounds and adding strategic pauses. If too slow, practice reducing your time by 15% while maintaining clarity. Repeat until comfortable."
            })
        elif weakest_area == 'accuracy':
            recommendations.append({
                'category': 'practice',
                'type': 'improvement',
                'text': "FOCUSED PRACTICE EXERCISE: Select 5-10 challenging words or phrases from your recording. Practice them in isolation, exaggerating the pronunciation of difficult sounds. Then practice them in context at a slow pace, gradually increasing to normal speed while maintaining clear articulation."
            })
        
        # Add general practice recommendation
        recommendations.append({
            'category': 'general',
            'type': 'improvement',
            'text': "Regular practice is key to improvement. Try recording yourself daily for short 1-2 minute sessions focused on specific skills. Even these brief, targeted exercises can lead to significant improvements over time."
        })
        
        return recommendations
    
    def generate_charts_data(self):
        """Generate enhanced visualization data for charts."""
        charts_data = {}
        
        # Generate waveform with pause markers
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(self.audio_data, sr=self.sample_rate, alpha=0.6)
        
        # Add markers for pauses
        if self.features.get('pause_timestamps'):
            for start, end in self.features['pause_timestamps']:
                plt.axvspan(start, end, color='red', alpha=0.2)
                plt.axvline(x=start, color='red', linestyle='--', alpha=0.4)
                plt.axvline(x=end, color='red', linestyle='--', alpha=0.4)
        
        plt.title('Audio Waveform with Pause Markers')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        waveform_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate pitch contour with variation highlights
        plt.figure(figsize=(10, 4))
        
        # Plot pitch data
        f0 = self.features.get('f0', None)
        voiced_flag = self.features.get('voiced_flag', None)
        times = self.features.get('pitch_times', None)
        
        if f0 is not None and voiced_flag is not None and times is not None:
            # Plot the pitch contour
            plt.plot(times, f0, 'g-', alpha=0.4, label='Pitch')
            
            # Highlight voiced regions with stronger color
            plt.plot(times[voiced_flag], f0[voiced_flag], 'g-', alpha=0.8)
            
            # Highlight regions with high pitch variation
            segment_times = self.features.get('segment_times', [])
            segment_variations = self.features.get('segment_pitch_variations', [])
            
            if segment_times and segment_variations:
                # Normalize variations to determine highlight intensity
                max_var = max(segment_variations) if segment_variations else 1
                for i, (t, var) in enumerate(zip(segment_times, segment_variations)):
                    if i < len(segment_times) - 1:
                        # Determine color based on variation (higher variation = stronger highlighting)
                        alpha = min(0.8, max(0.1, var / max_var))
                        if var > np.mean(segment_variations) + np.std(segment_variations):
                            # Highlight high variation segments
                            plt.axvspan(t, segment_times[i+1], color='blue', alpha=alpha*0.3)
        
        plt.title('Pitch Contour with Variation Highlights')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pitch_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate volume/energy chart
        plt.figure(figsize=(10, 4))
        
        # Plot volume data
        volume_data = self.features.get('volume_data', None)
        volume_times = self.features.get('volume_times', None)
        
        if volume_data is not None and volume_times is not None:
            plt.plot(volume_times, volume_data, 'b-', alpha=0.8)
            
            # Highlight low volume regions (potential mumbling)
            if len(volume_data) > 0:
                threshold = np.mean(volume_data) * 0.7
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Low Volume Threshold')
                
                # Mark regions below threshold
                below_threshold = volume_data < threshold
                for start_idx, end_idx in _contiguous_regions(below_threshold):
                    if end_idx - start_idx > 10:  # Only mark extended regions
                        plt.axvspan(volume_times[start_idx], volume_times[min(end_idx, len(volume_times)-1)], 
                                  color='red', alpha=0.2)
        
        plt.title('Volume/Energy Over Time')
        plt.ylabel('Amplitude (RMS)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        volume_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate speaking rate visualization
        plt.figure(figsize=(10, 4))
        
        # Create speaking rate visualization based on context
        context_params = self.context_parameters.get(self.context, self.context_parameters['general'])
        ideal_min = context_params['min_wpm']
        ideal_max = context_params['max_wpm']
        actual_rate = self.features.get('speaking_rate', 0)
        
        # Create a simple bar chart comparing actual to ideal range
        plt.bar(0, actual_rate, width=0.6, color='blue', alpha=0.7, label='Your Speaking Rate')
        
        # Add a rectangle for the ideal range
        plt.axhspan(ideal_min, ideal_max, alpha=0.2, color='green', label=f'Ideal Range for {self.context.replace("_", " ").title()}')
        
        # Add text annotations
        plt.text(0, actual_rate + 5, f'{actual_rate:.1f} WPM', ha='center')
        plt.text(1.2, (ideal_min + ideal_max)/2, f'Ideal: {ideal_min}-{ideal_max} WPM', 
                va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))
        
        plt.title('Speaking Rate Analysis')
        plt.ylabel('Words Per Minute (WPM)')
        plt.xticks([0], ['Your Rate'])
        plt.ylim(0, max(actual_rate, ideal_max) * 1.2)  # Make room for text
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pace_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate radar chart data for scores
        charts_data = {
            'waveform': waveform_b64,
            'pitch_contour': pitch_b64,
            'volume': volume_b64,
            'pace': pace_b64,
            'scores': [
                self.scores['modulation'],
                self.scores['pauses'],
                self.scores['expression'],
                self.scores['accuracy'],
                self.scores['pace'],
                self.scores['overall']
            ]
        }
        
        return charts_data


def _contiguous_regions(condition):
    """Find contiguous regions in a boolean array (helper function for charting)."""
    # Find the indices of changes in "condition"
    d = np.diff(np.concatenate(([0], condition.astype(np.int32), [0])))
    
    # Start indices of runs of "True"
    starts = np.where(d > 0)[0]
    
    # End indices of runs of "True"
    ends = np.where(d < 0)[0]
    
    # Return start and end indices as pairs
    return zip(starts, ends)


def analyze_speech(audio_file_path, reference_text, context="general"):
    """Analyze a speech recording and return detailed feedback."""
    analyzer = SpeechAnalyzer()
    analyzer.load_audio(audio_file_path)
    
    # Transcribe the audio
    transcription = analyzer.transcribe_audio()
    
    # Set reference text and extract features
    analyzer.set_reference_text(reference_text)
    analyzer.extract_features()
    
    # Calculate scores with context
    scores = analyzer.calculate_scores(context)
    
    # Generate context-specific recommendations
    recommendations = analyzer.generate_recommendations(context)
    
    # Generate chart data
    charts_data = analyzer.generate_charts_data()
    
    # Return comprehensive analysis
    return {
        'scores': scores,
        'recommendations': recommendations,
        'transcription': transcription,
        'charts_data': charts_data,
        'context': context
    }


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


# Initialize module
if __name__ == "__main__":
    # This is for testing purposes
    try:
        import soundfile as sf
        print("Testing speech analyzer with context-specific feedback...")
        
        test_file = "test_audio.wav"
        reference = "This is a test sentence for speech analysis."
        test_context = "business_presentation"  # Test with a specific context
        
        # Check if file exists
        if not os.path.exists(test_file):
            print(f"Test file {test_file} not found. Creating a simple test file...")
            
            # Generate a simple sine wave as test audio
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Generate a signal with varying frequency to simulate speech
            frequencies = np.linspace(100, 400, len(t))
            audio_data = 0.5 * np.sin(2 * np.pi * frequencies * t)
            
            # Save as WAV file
            sf.write(test_file, audio_data, sample_rate)
            print(f"Created test audio file: {test_file}")
        
        print(f"Analyzing test audio file: {test_file} with context: {test_context}")
        results = analyze_speech(test_file, reference, test_context)
        
        # Convert NumPy types to Python native types for JSON serialization
        serializable_results = {
            'scores': {k: float(v) for k, v in results['scores'].items()},
            'recommendations': results['recommendations'],
            'transcription': results['transcription'],
            'context': results['context'],
            # Keep only essential chart data to avoid base64 clutter in output
            'charts_data': {'scores': [float(s) for s in results['charts_data']['scores']]}
        }
        
        print("\n=== TEST RESULTS ===\n")
        print(f"Context: {serializable_results['context']}")
        print("\nScores:")
        for key, value in serializable_results['scores'].items():
            print(f"  {key}: {value}")
            
        print("\nTranscription:")
        print(f"  \"{serializable_results['transcription']}\"")
        
        print("\nRecommendations (sample):")
        # Print just a few recommendations for brevity
        for i, rec in enumerate(serializable_results['recommendations'][:3]):
            print(f"  {i+1}. [{rec['category']} - {rec['type']}] {rec['text'][:100]}...")
            
        print("\nFull analysis completed successfully!")
        print(f"Generated {len(serializable_results['recommendations'])} recommendations")
        print(f"Generated {len(results['charts_data'])} visualizations")
        
    except Exception as e:
        print(f"Error testing speech analyzer: {str(e)}")
        import traceback
        traceback.print_exc()