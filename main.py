from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import google.generativeai as genai
import os
import time
import json
import scipy.io.wavfile as wav
import threading
from dotenv import load_dotenv

# --- Configuration & State ---
CONFIG_FILE = "config.json"

class ConfigManager:
    @staticmethod
    def load():
        default_config = {
            "dictionary": [],
            "tone": "Casual",
            "context": "",
            "forbiddenWords": []
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return {**default_config, **json.load(f)}
            except:
                pass
        return default_config

    @staticmethod
    def save(config):
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

# --- Services ---
# Load Environment Variables
if os.path.exists('.env.local'):
    load_dotenv('.env.local')
else:
    load_dotenv()

API_KEY = os.getenv("VITE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

class GeminiService:
    def __init__(self, api_key):
        if not api_key:
            self.model = None
        else:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025") # As requested
            except Exception as e:
                print(f"Gemini Config Error: {e}")
                self.model = None

    def translate(self, audio_data, sample_rate, config):
        if not self.model:
            return {"balti": "Error", "english": "API Key Missing"}

        try:
            filename = "temp_audio.wav"
            # Normalize and convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav.write(filename, sample_rate, audio_int16)

            audio_file = genai.upload_file(path=filename)
            
            # Format Dictionary and Config
            dict_str = "\n".join([f"- {d['balti']} -> {d['english']}" for d in config.get('dictionary', [])])
            forbidden_str = ", ".join(config.get('forbiddenWords', []))

            prompt = f"""
            You are an expert Balti-to-English translator.
            
            CONFIGURATION:
            - Context: {config.get('context', 'General')}
            - Tone: {config.get('tone', 'Casual')}
            - Blocked Words: {forbidden_str}
            
            CUSTOM DICTIONARY:
            {dict_str}
            
            TASK:
            1. Transcribe the Balti speech.
            2. Translate it to English.
            
            RESPONSE FORMAT (JSON ONLY):
            {{ "balti": "...", "english": "..." }}
            """

            response = self.model.generate_content([prompt, audio_file])
            
            try:
                audio_file.delete()
                os.remove(filename)
            except: 
                pass

            text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(text)

        except Exception as e:
            print(f"Translation Error: {e}")
            return {"balti": "Error", "english": "Translation Failed"}

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.sample_rate = 16000

    def start(self):
        self.recording = True
        self.audio_data = []
        try:
            self.stream = sd.InputStream(callback=self._callback, channels=1, samplerate=self.sample_rate)
            self.stream.start()
            print("Recording started...")
        except Exception as e:
            print(f"Mic Error: {e}")
            self.recording = False

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Recording stopped.")
        return np.concatenate(self.audio_data, axis=0) if self.audio_data else np.array([])

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# Initialize Services
gemini = GeminiService(API_KEY)
recorder = AudioRecorder()
config = ConfigManager.load()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global config
    if request.method == 'POST':
        config = request.json
        ConfigManager.save(config)
        return jsonify({"status": "saved"})
    return jsonify(config)

@app.route('/api/record', methods=['POST'])
def handle_record():
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        recorder.start()
        return jsonify({"status": "started"})
    
    elif action == 'stop':
        audio = recorder.stop()
        if len(audio) == 0:
             return jsonify({"balti": "Error", "english": "No audio captured"})
             
        result = gemini.translate(audio, 16000, config)
        return jsonify(result)

if __name__ == "__main__":
    print("Starting Balti Voice AI Web Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(port=5000, debug=True)
