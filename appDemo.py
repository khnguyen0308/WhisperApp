# app.py
import re
import streamlit as st
from openai import OpenAI
from iso_639_languages import iso_639_languages
from pydub import AudioSegment
import tempfile
import os
import warnings
from pydub.utils import which
from difflib import SequenceMatcher
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter

warnings.filterwarnings("ignore", category=SyntaxWarning)

AudioSegment.converter = which("ffmpeg")
st.set_page_config(layout="wide")

def preprocess_audio(file_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(16000)  # preferred for Whisper
    audio = normalize(audio)  # volume normalization
    audio = high_pass_filter(audio, cutoff=100)  # cut hum
    audio = low_pass_filter(audio, cutoff=8000)  # cut hiss
    return audio

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[How to get an OpenAI API key?](https://platform.openai.com/account/api-keys)"

st.header('Whisper WebUI', divider='violet')
st.caption('')

st.subheader("What do you want Whisper to do for you?", divider=True)
usecase_option = st.selectbox(
    "transcription or translation",
    ("Create transcription", "Create translation"),
)

st.subheader("Audio file", divider=True)
audio_file = st.file_uploader('Choose an audio file', type=["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"])

st.subheader("Option customization", divider=True)
if usecase_option == "Create transcription":
    col1, col2, col3 = st.columns(3)

    with col1:
        language_option = st.selectbox(
            "Input Language (Optional):",
            iso_639_languages.keys()
        )
        st.caption("The language of the input audio. ISO-639-1 format helps accuracy and latency.")
        language_code = iso_639_languages[language_option]

    with col2:
        prompt = st.text_input("Prompt (Optional)", "")

    with col3:
        format_option = st.selectbox(
            "Output Format (Optional):",
            ['text', 'srt', 'vtt']
        )

    if audio_file:
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        if st.button('Transcribe Audio'):
            client = OpenAI(api_key=openai_api_key)
            with st.spinner('Processing...'):
                transcription_parts = []
                file_size_mb = audio_file.size / (1024 * 1024)

                if file_size_mb <= 25:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language_code,
                        prompt=prompt,
                        response_format=format_option,
                    )
                    transcription_parts.append(transcription)
                else:
                    # Split and process the file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name

                    song = preprocess_audio(tmp_path)

                    chunk_length_ms = 10 * 60 * 1000
                    overlap_ms = 2 * 1000 
                    chunks = [song[i:i + chunk_length_ms + overlap_ms] for i in range(0, len(song), chunk_length_ms)]

                    for idx, chunk in enumerate(chunks):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as chunk_file:
                            chunk.export(chunk_file.name, format="mp3")
                            chunk_path = chunk_file.name

                        with open(chunk_path, "rb") as f:
                            response = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language=language_code,
                                prompt=prompt,
                                response_format="text",
                            )
                            if format_option == "text":
                                transcription_parts.append(response.strip())
                            else:
                                transcription_parts.append(str(response))

                        os.unlink(chunk_path)
                    os.unlink(tmp_path)

                full_transcription = "\n".join(transcription_parts)
    
                if format_option in ['text', 'srt', 'vtt']:
                    st.text_area("Transcription Output", value=full_transcription, height=400)
                else:
                    st.write("Note: JSON output format is not supported when using multiple chunks.")

elif usecase_option == "Create translation":
    col1, col2 = st.columns(2)

    with col1:
        prompt = st.text_input("Prompt (Optional)", "")

    with col2:
        format_option = st.selectbox(
            "Output Format (Optional):",
            ['text', 'srt', 'vtt']
        )

    if audio_file:
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        if st.button('Translation Audio'):
            client = OpenAI(api_key=openai_api_key)
            with st.spinner('Processing...'):
                translation = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    prompt=prompt,
                    response_format=format_option,
                )

                if format_option == 'json':
                    st.json(translation.to_json())
                elif format_option == 'text':
                    st.text_area("Translation Output", value=str(translation), height=400)
                elif format_option == 'verbose_json':
                    st.json(translation.to_json())
                elif format_option == 'vtt':
                    st.text_area("VTT Output", value=str(translation), height=400)
                elif format_option == 'srt':
                    st.download_button(label='Click To Download SRT File', data=str(translation), file_name=audio_file.name + '.srt')
