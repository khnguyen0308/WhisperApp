from flask import Flask, request, jsonify, render_template, send_file
import os
import tempfile
from model import transcribe_audio, convert_to_srt, convert_to_vtt
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Define a persistent directory for storing transcription files
PERSISTENT_DIR = '/home/site/transcriptions'
os.makedirs(PERSISTENT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio-file']
    language = request.form.get('language', 'vi')  # Default to Vietnamese
    output_format = request.form.get('output-format', 'text')  # Default to text
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Generate a unique identifier for the file
        unique_id = str(uuid.uuid4())
        original_filename = os.path.splitext(audio_file.filename)[0]  # Get the base name without extension
        unique_filename = f"{original_filename}_{unique_id}.mp3"

        # Save the uploaded file temporarily
        audio_file_path = os.path.join(PERSISTENT_DIR, unique_filename)
        audio_file.save(audio_file_path)
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Failed to save audio file at '{audio_file_path}'")

        # Transcribe the audio file
        transcription_text = transcribe_audio(audio_file_path, language)

        # Save transcription to the requested format in the persistent directory
        transcription_filename = f"{original_filename}_{unique_id}.{output_format}"
        transcription_file_path = os.path.join(PERSISTENT_DIR, transcription_filename)
        with open(transcription_file_path, 'w', encoding='utf-8') as f:
            if output_format == 'srt':
                f.write(convert_to_srt(transcription_text))
            elif output_format == 'vtt':
                f.write(convert_to_vtt(transcription_text))
            else:
                f.write(transcription_text)

        return jsonify({
            'transcription': transcription_text,
            'download_url': f'/download/{transcription_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(PERSISTENT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)