import openai
import os
import tempfile
import logging
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
from pydub.utils import which
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential # Import tenacity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info(f"Using endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")

# Load environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = "2024-06-01"

# Model and deployment configuration
model_name = "whisper"
deployment_id = "whisper"

# Set ffmpeg path for pydub - handle different environments
try:
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
    else:
        logger.warning("FFmpeg not found in PATH")
except Exception as e:
    logger.error(f"Error setting up FFmpeg: {e}")

# Preprocessing function with better error handling
def preprocess_audio(file_path: str) -> AudioSegment:
    """Preprocess audio file with error handling"""
    try:
        logger.info(f"Preprocessing audio file: {file_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read audio file: {file_path}")
        
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        logger.info(f"Original audio: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
        
        # Preprocessing steps
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set preferred frame rate
        audio = normalize(audio)  # Normalize volume
        audio = high_pass_filter(audio, cutoff=100)  # Remove low-frequency noise
        audio = low_pass_filter(audio, cutoff=8000)  # Remove high-frequency noise
        
        logger.info(f"Preprocessed audio: {len(audio)}ms")
        return audio
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise RuntimeError(f"Failed to preprocess audio file: {e}")

# Helper function to convert transcription to SRT format
def convert_to_srt(transcription: str) -> str:
    """Convert transcription text to SRT format"""
    lines = [line.strip() for line in transcription.split("\n") if line.strip()]
    srt_output = []
    
    for idx, line in enumerate(lines, start=1):
        # Calculate timing (rough estimate - 3 seconds per line)
        start_time = (idx - 1) * 3
        end_time = idx * 3
        
        start_hours = start_time // 3600
        start_minutes = (start_time % 3600) // 60
        start_seconds = start_time % 60
        
        end_hours = end_time // 3600
        end_minutes = (end_time % 3600) // 60
        end_seconds = end_time % 60
        
        srt_entry = f"{idx}\n{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},000 --> {end_hours:02d}:{end_minutes:02d}:{end_seconds:02d},000\n{line}\n"
        srt_output.append(srt_entry)
    
    return "\n".join(srt_output)

# Helper function to convert transcription to VTT format
def convert_to_vtt(transcription: str) -> str:
    """Convert transcription text to VTT format"""
    srt_content = convert_to_srt(transcription)
    return "WEBVTT\n\n" + srt_content

# Main transcription function with improved error handling
def transcribe_audio(audio_file_path: str, language: str = "vi", output_format: str = "text") -> str:
    """
    Transcribe audio file with proper error handling and path management
    """
    # Use absolute path to ensure consistency
    abs_audio_path = os.path.abspath(audio_file_path)
    logger.info(f"Starting transcription for: {abs_audio_path}")
    logger.info(f"Language: {language}, Format: {output_format}")
    
    # Verify file exists
    if not os.path.exists(abs_audio_path):
        error_msg = f"Audio file not found at '{abs_audio_path}'"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check file permissions
    if not os.access(abs_audio_path, os.R_OK):
        error_msg = f"Cannot read audio file at '{abs_audio_path}'"
        logger.error(error_msg)
        raise PermissionError(error_msg)
    
    try:
        # Check file size
        file_size_bytes = os.path.getsize(abs_audio_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        if file_size_mb <= 25:
            # Process small files directly
            logger.info("Processing file directly (≤25MB)")
            transcription_text = transcribe_single_file(abs_audio_path, language)
        else:
            # Process large files in chunks
            logger.info("Processing large file in chunks (>25MB)")
            transcription_text = transcribe_large_file(abs_audio_path, language)

        # Convert transcription to the requested format
        if output_format == "srt":
            return convert_to_srt(transcription_text)
        elif output_format == "vtt":
            return convert_to_vtt(transcription_text)
        else:
            return transcription_text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"An error occurred during transcription: {e}")

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60), # Wait 4s, then 8s, 16s, etc.
    stop=stop_after_attempt(5), # Stop after 5 attempts
    reraise=True # Reraise the exception if all retries fail
)
def transcribe_chunk_with_retry(chunk_audio, language):
    """A wrapper for the API call with retry logic."""
    logger.info("Attempting to transcribe chunk...")
    result = openai.Audio.transcribe(
        file=chunk_audio,
        model=model_name,
        deployment_id=deployment_id,
        language=language
    )
    return result["text"]

def transcribe_single_file(file_path: str, language: str) -> str:
    """Transcribe a single audio file with retry logic"""
    try:
        with open(file_path, "rb") as audio_file:
            # You can also apply the retry logic here if single files could fail
            # For now, we apply it to the chunking logic which is the main cause
            logger.info("Transcribing single file...")
            result = openai.Audio.transcribe(
                file=audio_file,
                model=model_name,
                deployment_id=deployment_id,
                language=language
            )
            return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing single file: {e}")
        raise

def transcribe_large_file(file_path: str, language: str) -> str:
    """Transcribe a large audio file by splitting into chunks with retry logic"""
    temp_files = []  # Keep track of temporary files for cleanup
    
    try:
        # Preprocess the audio
        audio = preprocess_audio(file_path)
        
        # Define chunk parameters
        chunk_length_ms = 10 * 60 * 1000  # 10 minutes
        overlap_ms = 2 * 1000  # 2 seconds overlap
        
        # Calculate chunks
        total_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms
        logger.info(f"Splitting into {total_chunks} chunks")
        
        transcription_parts = []
        
        for idx in range(0, len(audio), chunk_length_ms):
            chunk_start = max(0, idx - overlap_ms if idx > 0 else 0)
            chunk_end = min(len(audio), idx + chunk_length_ms + overlap_ms)
            chunk = audio[chunk_start:chunk_end]
            
            # Create temporary file for chunk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tempfile.gettempdir()) as chunk_file:
                chunk_path = chunk_file.name
                temp_files.append(chunk_path)
                
                logger.info(f"Exporting chunk {len(temp_files)}/{total_chunks} to: {chunk_path}")
                chunk.export(chunk_path, format="wav")
                
                # Verify chunk file was created
                if not os.path.exists(chunk_path):
                    raise RuntimeError(f"Failed to create chunk file: {chunk_path}")
                
                # Transcribe chunk
                with open(chunk_path, "rb") as chunk_audio:
                    # MODIFIED LINE
                    transcription_text = transcribe_chunk_with_retry(chunk_audio, language)
                    transcription_parts.append(transcription_text)
                    
                    logger.info(f"Completed chunk {len(temp_files)}/{total_chunks}")
        
        # Combine all transcription parts
        final_transcription = " ".join(transcription_parts)
        logger.info("Large file transcription completed successfully")
        return final_transcription
        
    except Exception as e:
        logger.error(f"Error transcribing large file: {e}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {cleanup_error}")

# Test function to verify setup
def test_transcription_setup():
    """Test function to verify the transcription setup"""
    try:
        logger.info("Testing transcription setup...")
        
        # Check environment variables
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        # Check FFmpeg
        if not which("ffmpeg"):
            logger.warning("FFmpeg not found - audio processing may fail")
        
        logger.info("Transcription setup test completed")
        return True
        
    except Exception as e:
        logger.error(f"Setup test failed: {e}")
        return False

if __name__ == "__main__":
    test_transcription_setup()