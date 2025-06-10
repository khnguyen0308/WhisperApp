from flask import Flask, request, jsonify, render_template, send_file
import os
import tempfile
from model import transcribe_audio, convert_to_srt, convert_to_vtt
import time
import uuid
from dotenv import load_dotenv
import logging
import threading ## MODIFICATION: Import the threading module

load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a persistent directory for storing transcription files
if os.environ.get('WEBSITE_SITE_NAME'):
    PERSISTENT_DIR = '/home/site/transcriptions'
else:
    PERSISTENT_DIR = os.path.join(os.getcwd(), 'transcriptions')

os.makedirs(PERSISTENT_DIR, exist_ok=True)
logger.info(f"Using transcription directory: {PERSISTENT_DIR}")

## MODIFICATION: Create a dictionary to store the status and results of our tasks.
## In a real-world multi-worker scenario, this should be replaced with a more robust
## solution like Redis or a database, but this will work for a single instance.
tasks = {}

## MODIFICATION: This new function will run our transcription in the background.
def run_transcription_in_background(audio_file_path, language, output_format, task_id):
    """
    A wrapper function to run the transcription process in a background thread
    and update the global 'tasks' dictionary with the result.
    """
    logger.info(f"Background task {task_id} started for {audio_file_path}.")
    try:
        # Get the original filename without extension for the output file
        original_filename = os.path.splitext(os.path.basename(audio_file_path))[0].split('_')[0]

        # Perform the transcription
        transcription_text = transcribe_audio(audio_file_path, language)

        # Determine the output file extension based on the requested format
        if output_format == 'srt':
            output_content = convert_to_srt(transcription_text)
            file_extension = 'srt'
        elif output_format == 'vtt':
            output_content = convert_to_vtt(transcription_text)
            file_extension = 'vtt'
        else:
            output_content = transcription_text
            file_extension = 'txt'
        
        # Create a unique filename for the transcription output
        transcription_filename = f"{original_filename}_{task_id}.{file_extension}"
        transcription_file_path = os.path.join(PERSISTENT_DIR, transcription_filename)
        
        # Save the transcription output
        with open(transcription_file_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"Transcription for task {task_id} completed successfully.")
        
        # Update the task status to 'completed' with the result
        tasks[task_id] = {
            'status': 'completed',
            'transcription': transcription_text,
            'download_url': f'/download/{transcription_filename}'
        }

    except Exception as e:
        logger.error(f"Error during background transcription for task {task_id}: {str(e)}")
        # Update the task status to 'failed'
        tasks[task_id] = {'status': 'failed', 'error': str(e)}
    finally:
        # Clean up the original audio file after processing is complete
        if os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
                logger.info(f"Cleaned up audio file for task {task_id}: {audio_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup audio file for task {task_id}: {cleanup_error}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio-file']
    language = request.form.get('language', 'vi')
    output_format = request.form.get('output-format', 'text')
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    ## MODIFICATION: Use the task_id as the unique identifier for everything
    task_id = str(uuid.uuid4())
    original_filename = os.path.splitext(audio_file.filename)[0]
    
    # Create a unique filename for the audio file using the task_id
    audio_extension = os.path.splitext(audio_file.filename)[1]
    unique_filename = f"{original_filename}_{task_id}{audio_extension}"
    audio_file_path = os.path.join(PERSISTENT_DIR, unique_filename)
    
    try:
        logger.info(f"Saving audio file to: {audio_file_path}")
        audio_file.save(audio_file_path)
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Failed to save audio file at '{audio_file_path}'")

        # Mark the task as 'processing'
        tasks[task_id] = {'status': 'processing'}

        ## MODIFICATION: Start the transcription in a background thread.
        ## We pass all necessary info to the background function.
        thread = threading.Thread(
            target=run_transcription_in_background,
            args=(os.path.abspath(audio_file_path), language, output_format, task_id)
        )
        thread.daemon = True  # Allows main app to exit even if threads are running
        thread.start()
        
        logger.info(f"Task {task_id} started for {audio_file_path}")
        
        ## MODIFICATION: Immediately return a response to the user.
        ## This response includes the task_id and a URL to check the status.
        return jsonify({
            'message': 'Transcription has started. Please check the status URL for the result.',
            'task_id': task_id,
            'status_url': f'/status/{task_id}'
        }), 202  # 202 Accepted status code is appropriate here

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

## MODIFICATION: Add a new endpoint for checking the status of a task.
@app.route('/status/<task_id>')
def status(task_id):
    """Endpoint to check the status of a transcription task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task)

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(PERSISTENT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        logger.error(f"Download file not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'transcription_dir': PERSISTENT_DIR,
        'dir_exists': os.path.exists(PERSISTENT_DIR)
    })

if __name__ == '__main__':
    app.run(debug=True)