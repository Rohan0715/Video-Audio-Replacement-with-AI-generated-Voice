import streamlit as st
import torch
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from gtts import gTTS
import tempfile
import requests
import time

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

st.title("Video Audio Replacement with AI-generated Voice")
st.header("Step 1: Upload a video file")
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov"])

def slow_down_audio_properly(audio_path, output_path, slowdown_factor=0.80):
    audio = AudioSegment.from_file(audio_path)
    new_frame_rate = int(audio.frame_rate * slowdown_factor)
    slowed_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
    slowed_audio = slowed_audio.set_frame_rate(audio.frame_rate)
    slowed_audio.export(output_path, format="wav")
    return output_path

def split_audio(audio_file, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(audio_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks

def extract_audio(video_file):
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_with_wav2vec2(audio_file):
    speech, sr = librosa.load(audio_file, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

azure_openai_key = ""
azure_openai_endpoint = r""

def correct_transcription_with_openai(transcribed_text):
    headers = {"Content-Type": "application/json", "api-key": azure_openai_key}
    data = {
"messages": [{
    "role": "user", 
    "content": f"Correct the grammar and clarity of this text without adding any extra comments or explanations: {transcribed_text}. Ensure that every new line is retained as a clear pause. If there’s nothing to convert, include a placeholder like '.' to denote a pause for synchronization.Also the corrected line words of each line should'nt be much greater then wrong line cause i want it syncronize the correct statement in place of wrong statement "}],
        "max_tokens": 1000
    }
    try:
        response = requests.post(azure_openai_endpoint, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            corrected_text = result["choices"][0]["message"]["content"].strip()
            return corrected_text
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_audio_from_text(text):
    language = "en"
    sound = gTTS(text=text, lang=language, slow=False, tld="com.au")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_path = temp_audio.name
    sound.save(audio_path)
    audio = AudioSegment.from_mp3(audio_path)
    new_audio = audio.speedup(playback_speed=1.22)
    modified_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    new_audio.export(modified_audio_path, format="mp3")
    return modified_audio_path

def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    video = video.set_audio(new_audio)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

if uploaded_video is not None:
    st.success("✅ Video uploaded successfully!")
    audio_path = extract_audio(uploaded_video)
    st.video(uploaded_video)
    
    with st.spinner("🔊 Extracting audio from the video..."):
        time.sleep(2)  # Simulate delay for audio extraction
    

    with st.spinner("⏳ Slowing down the audio for better transcription..."):
        slowed_audio_path = "slowed_audio.wav"
        slow_down_audio_properly(audio_path, slowed_audio_path, slowdown_factor=0.75)
    
    st.success(f"✅ Audio slowed down successfully! Saved as {slowed_audio_path}")
    with open(slowed_audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    
    st.audio(audio_bytes, format="audio/wav")
    st.info("📝 Transcribing slowed down audio to text...")
    
    with st.spinner("🔂 Splitting the slowed down audio into smaller chunks..."):
        audio_chunks = split_audio(slowed_audio_path)
    full_transcription = ""
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(audio_chunks):
        st.info(f"📝 Transcribing chunk: {chunk}")
        with st.spinner(f"🔄 Transcribing chunk {idx+1}/{len(audio_chunks)}..."):
            chunk_transcription = transcribe_audio_with_wav2vec2(chunk)
            full_transcription += chunk_transcription + "\n"
        progress_bar.progress((idx + 1) / len(audio_chunks))
    with st.spinner("🔍 Correcting transcription using Azure OpenAI..."):
        corrected_transcription = correct_transcription_with_openai(full_transcription)
    
    # Display the transcriptions
    st.text_area("Full Transcription", full_transcription, height=200)
    st.text_area("Corrected Transcription", corrected_transcription, height=200)
    
    if corrected_transcription:
        # Generate corrected audio from transcription
        with st.spinner("🔉 Generating corrected audio from transcription..."):
            corrected_audio_path = generate_audio_from_text(corrected_transcription)
        
        st.success("✅ Corrected Audio Generated")
        st.audio(corrected_audio_path, format="audio/mp3")
        
        # Replace the original audio in the video
        with st.spinner("🔄 Replacing audio in the video..."):
            output_video_path = "output_video.mp4"
            replace_audio_in_video("uploaded_video.mp4", corrected_audio_path, output_video_path)
        
        st.success(f"✅ Video with corrected audio saved as {output_video_path}")
        st.video(output_video_path)