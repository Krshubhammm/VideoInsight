import os
import tempfile
from moviepy.editor import VideoFileClip
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import json
import time

load_dotenv()

VIDEO_PATH = r"video\sample.mp4"

class APIKeyManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.usage = {key: 0 for key in self.api_keys}

    def _load_api_keys(self):
        keys = os.environ.get("GROQ_API_KEYS", "").split(",")
        return [key.strip() for key in keys if key.strip()]

    def get_current_key(self):
        return self.api_keys[self.current_key_index]

    def rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    def update_usage(self, key, amount):
        self.usage[key] += amount
        if self.usage[key] >= 90:  # 90% usage threshold
            self.rotate_key()

api_manager = APIKeyManager()

def get_groq_client():
    return OpenAI(
        api_key=api_manager.get_current_key(),
        base_url="https://api.groq.com/openai/v1"
    )

def convert_video_to_audio(video_path):
    audio_path = video_path.replace('.mp4', '.mp3')
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

def chunk_audio(audio_path, chunk_length_ms=600000):
    audio = AudioSegment.from_mp3(audio_path)
    return [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

def transcribe_audio_chunk(audio_chunk):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        audio_chunk.export(temp_file.name, format="mp3")
        with open(temp_file.name, "rb") as audio_file:
            while True:
                try:
                    groq = get_groq_client()
                    transcript = groq.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        response_format="json"
                    )
                    api_manager.update_usage(api_manager.get_current_key(), 10)  # Assume 10% usage per request
                    break
                except Exception as e:
                    if "API key" in str(e):
                        print(f"API key error: {e}. Rotating to next key.")
                        api_manager.rotate_key()
                    else:
                        print(f"Error: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
    os.unlink(temp_file.name)
    return json.loads(transcript.model_dump_json())

def json_to_vtt(json_transcript, time_offset=0):
    vtt = "WEBVTT\n\n"
    if 'text' in json_transcript:
        start = ms_to_time(time_offset)
        end = ms_to_time(time_offset + 10000)
        text = json_transcript['text'].strip()
        vtt += f"{start} --> {end}\n{text}\n\n"
    return vtt

def ms_to_time(ms):
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def process_video(video_path):
    print("Converting video to audio...")
    audio_path = convert_video_to_audio(video_path)
    print("Chunking audio and transcribing...")
    audio_chunks = chunk_audio(audio_path)
    vtt_contents = []
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i+1}/{len(audio_chunks)}...")
        json_transcript = transcribe_audio_chunk(chunk)
        vtt_content = json_to_vtt(json_transcript, i * 600000)
        vtt_contents.append(vtt_content)
    combined_vtt = "".join(vtt_contents)
    os.remove(audio_path)
    return combined_vtt

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: The file {VIDEO_PATH} does not exist.")
        return
    
    vtt_content = process_video(VIDEO_PATH)
    output_path = os.path.splitext(VIDEO_PATH)[0] + ".vtt"
    with open(output_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write(vtt_content)
    print(f"\nVTT file saved as: {output_path}")

if __name__ == "__main__":
    main()