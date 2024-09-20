import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import wget
import gdown
import subprocess
import shutil
from scipy.io import wavfile
import json
import argparse
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import instaloader
from moviepy.editor import VideoFileClip

# Folder structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "voice_clone_data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MODEL_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
REEL_DIR = os.path.join(DATA_DIR, "reels")

# Create necessary directories
for dir in [DATA_DIR, AUDIO_DIR, MODEL_DIR, OUTPUT_DIR, REEL_DIR]:
    os.makedirs(dir, exist_ok=True)

# Download pre-trained model
def download_pretrained_model():
    model_path = os.path.join(MODEL_DIR, "tortoise-tts.zip")
    if not os.path.exists(model_path):
        print("Downloading pre-trained VITS model...")
        gdown.download("https://drive.google.com/uc?id=1t4MFpGlz-uvGM7r1F5O1Xgn7vFmEvnTM", model_path, quiet=False)
        
        print("Extracting model files...")
        shutil.unpack_archive(model_path, MODEL_DIR)
        os.remove(model_path)
    else:
        print("Pre-trained model already downloaded.")

# Instagram scraping function
def scrape_reels(username):
    L = instaloader.Instaloader()
    profile = instaloader.Profile.from_username(L.context, username)
    
    reels = []
    print(f"Scraping reels from {username}...")
    for post in tqdm(profile.get_posts()):
        if post.is_video:
            L.download_post(post, target=REEL_DIR)
            reels.append(post.video_url)
    
    return reels

# Video to audio conversion function
def convert_to_audio(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, verbose=False, logger=None)
    video.close()

# Audio preprocessing
def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    return audio, sr

# Voice cloning function
def clone_voice(audio_files):
    print("Preparing voice for cloning...")
    voice_samples = [load_audio(file, 22050) for file in audio_files]
    return voice_samples

# Text-to-Speech function
def text_to_speech(text, voice_samples, output_path):
    print(f"Converting text to speech: {text}")
    tts = TextToSpeech()
    gen_audio = tts.tts_with_preset(text, voice_samples=voice_samples, preset='fast')
    torchaudio.save(output_path, gen_audio.squeeze(0).cpu(), 24000)

# Main function
def main(instagram_username):
    # Download pre-trained model
    download_pretrained_model()

    # Scrape reels
    scrape_reels(instagram_username)

    # Convert videos to audio
    video_files = [os.path.join(REEL_DIR, f) for f in os.listdir(REEL_DIR) if f.endswith('.mp4')]
    audio_files = []
    print("Converting videos to audio...")
    for i, video_file in enumerate(tqdm(video_files)):
        audio_path = os.path.join(AUDIO_DIR, f"{instagram_username}_{i+1}.wav")
        convert_to_audio(video_file, audio_path)
        audio_files.append(audio_path)

    # Clone voice
    voice_samples = clone_voice(audio_files)

    # Generate TTS output
    output_path = os.path.join(OUTPUT_DIR, f"{instagram_username}_tts_output.wav")
    text_to_speech("Hello, this is a test of the cloned voice.", voice_samples, output_path)

    print(f"Voice cloning and TTS complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instagram Voice Cloning and TTS Script")
    parser.add_argument("username", help="Instagram username for the target voice")
    args = parser.parse_args()

    main(args.username)
