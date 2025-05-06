from flask import Flask, request, jsonify, render_template, send_file
import os
import tempfile
from model import transcribe_audio, convert_to_srt, convert_to_vtt
import time

app = Flask(__name__)
# Define a persistent directory for storing transcription files
PERSISTENT_DIR = os.path.join(os.getenv('HOME', '/home'), 'site', 'transcriptions')
os.makedirs(PERSISTENT_DIR, exist_ok=True)  # Ensure the directory exists

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
# def cleanup_old_files(directory, max_age_seconds=86400):  # 24 hours
#     now = time.time()
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_seconds:
#             os.remove(file_path)

def transcribe():
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio-file']
    language = request.form.get('language', 'vi')  # Default to Vietnamese
    output_format = request.form.get('output-format', 'text')  # Default to text
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file temporarily
        original_filename = os.path.splitext(audio_file.filename)[0]  # Get the base name without extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)
            audio_file_path = temp_audio.name

        # Transcribe the audio file
        transcription_text = transcribe_audio(audio_file_path, language)

        # Save transcription to the requested format in the persistent directory
        transcription_filename = f"{original_filename}.{output_format}"
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