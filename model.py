import openai
import os
import tempfile
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
from pydub.utils import which
from dotenv import load_dotenv

load_dotenv()
print("Using endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))

# Load environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Example: https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = "azure"
openai.api_version = "2024-06-01"

# Model and deployment configuration
model_name = "whisper"
deployment_id = "whisper"  # Replace with the correct deployment name

# Set ffmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

# Preprocessing function
def preprocess_audio(file_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set preferred frame rate
    audio = normalize(audio)  # Normalize volume
    audio = high_pass_filter(audio, cutoff=100)  # Remove low-frequency noise
    audio = low_pass_filter(audio, cutoff=8000)  # Remove high-frequency noise
    return audio

# Helper function to convert transcription to SRT format
def convert_to_srt(transcription: str) -> str:
    lines = transcription.split("\n")
    srt_output = []
    for idx, line in enumerate(lines, start=1):
        srt_output.append(f"{idx}\n00:00:{idx:02d},000 --> 00:00:{idx + 1:02d},000\n{line}\n")
    return "\n".join(srt_output)

# Helper function to convert transcription to VTT format
def convert_to_vtt(transcription: str) -> str:
    srt_content = convert_to_srt(transcription)
    return "WEBVTT\n\n" + srt_content

# Transcription function
def transcribe_audio(audio_file_path: str, language: str = "vi", output_format: str = "text") -> str:
    try:
        # Check file size
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)

        if file_size_mb <= 25:
            # Process small files directly
            result = openai.Audio.transcribe(
                file=open(audio_file_path, "rb"),
                model=model_name,
                deployment_id=deployment_id,
                language=language  # Use the selected language
            )
            transcription_text = result["text"]
        else:
            # Process large files in chunks
            print("Audio file is larger than 25MB. Splitting into chunks...")
            audio = preprocess_audio(audio_file_path)
            
            # Define chunk size and overlap
            chunk_length_ms = 10 * 60 * 1000  # 10 minutes
            overlap_ms = 2 * 1000  # 2 seconds overlap
            chunks = [audio[i:i + chunk_length_ms + overlap_ms] for i in range(0, len(audio), chunk_length_ms)]

            transcription_parts = []
            for idx, chunk in enumerate(chunks):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                    chunk.export(chunk_file.name, format="wav")
                    chunk_path = chunk_file.name

                result = openai.Audio.transcribe(
                    file=open(chunk_path, "rb"),
                    model=model_name,
                    deployment_id=deployment_id,
                    language=language  # Use the selected language
                )
                transcription_parts.append(result["text"])

                # Clean up temporary file
                os.unlink(chunk_path)
            transcription_text = "\n".join(transcription_parts)

        # Convert transcription to the requested format
        if output_format == "srt":
            return convert_to_srt(transcription_text)
        elif output_format == "vtt":
            return convert_to_vtt(transcription_text)
        else:
            return transcription_text

    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found at '{audio_file_path}'")
    except Exception as e:
        raise RuntimeError(f"An error occurred during transcription: {e}")