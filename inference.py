import os
import torch
import cv2
import numpy as np
import torchaudio
import subprocess
import whisper
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

MODEL_PATH = "saved_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

# Mappings
EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def split_video(input_path):
    """
    Splits video based on duration rules:
    1. < 10s: One clip.
    2. 10s <= duration < 15s: Split into 2 equal halves.
    3. >= 15s: Split into 10s chunks.
    """
    clip = VideoFileClip(input_path)
    duration = clip.duration
    segments = []
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if duration < 10:
        out_name = f"temp_{base_name}_full.mp4"
        clip.write_videofile(out_name, codec="libx264",
                             audio_codec="aac", verbose=False, logger=None)
        segments.append((0, duration, out_name))

    elif 10 <= duration < 15:
        mid_point = duration / 2
        # Part 1
        out_1 = f"temp_{base_name}_part1.mp4"
        clip.subclip(0, mid_point).write_videofile(
            out_1, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        segments.append((0, mid_point, out_1))
        # Part 2
        out_2 = f"temp_{base_name}_part2.mp4"
        clip.subclip(mid_point, duration).write_videofile(
            out_2, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        segments.append((mid_point, duration, out_2))

    else:
        current_time = 0
        part_idx = 1
        while current_time < duration:
            end_time = min(current_time + 10, duration)
            out_name = f"temp_{base_name}_part{part_idx}.mp4"
            clip.subclip(current_time, end_time).write_videofile(
                out_name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            segments.append((current_time, end_time, out_name))
            current_time += 10
            part_idx += 1

    clip.close()
    return segments


def extract_features(video_path, text_query):
    # 1. Text (BERT)
    text_inputs = TOKENIZER(text_query, padding="max_length",
                            truncation=True, max_length=128, return_tensors='pt')

    # 2. Video (Frames)
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while len(frames) < 30 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3))]
    if len(frames) < 30:
        frames += [np.zeros_like(frames[0])] * (30 - len(frames))
    frames = frames[:30]
    video_tensor = torch.FloatTensor(
        np.array(frames)).permute(0, 3, 1, 2).unsqueeze(0)

    # 3. Audio (MelSpec)
    audio_path = video_path.replace(".mp4", ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    waveform, sr = torchaudio.load(audio_path)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)(waveform)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

    if mel_spec.size(2) < 300:
        mel_spec = torch.nn.functional.pad(
            mel_spec, (0, 300 - mel_spec.size(2)))
    else:
        mel_spec = mel_spec[:, :, :300]

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return text_inputs, video_tensor, mel_spec.unsqueeze(0)


def transcribe_with_whisper(video_path):
    """
    Uses OpenAI Whisper locally to transcribe video audio.
    """
    try:
        # Whisper handles audio extraction internally or accepts video files directly
        result = whisper_model.transcribe(video_path)
        text = result["text"].strip()
        return text if text else "unknown"
    except Exception as e:
        print(f"Whisper Error: {e}")
        return "unknown"


def analyze_video(raw_video_path):
    print(f"\n--- Processing Video: {raw_video_path} ---")

    # Load Your Trained Model
    try:
        model = MultimodalSentimentModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    segments = split_video(raw_video_path)
    full_results = []

    for start_t, end_t, clip_path in segments:
        print(f"Analyzing segment: {start_t:.1f}s to {end_t:.1f}s...")

        # 1. Transcribe (Local Whisper)
        transcript = transcribe_with_whisper(clip_path)

        # 2. Extract Features
        text_in, vid_in, aud_in = extract_features(clip_path, transcript)

        # 3. Predict
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in text_in.items()}
            vid_in = vid_in.to(DEVICE)
            aud_in = aud_in.to(DEVICE)

            outputs = model(inputs, vid_in, aud_in)
            emo_probs = torch.softmax(outputs['emotions'], dim=1)[0]
            sent_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

        # Rank Emotions
        ranked_emotions = [(EMOTION_MAP[i], p.item())
                           for i, p in enumerate(emo_probs)]
        ranked_emotions.sort(key=lambda x: x[1], reverse=True)

        # Rank Sentiments
        ranked_sentiments = [(SENTIMENT_MAP[i], p.item())
                             for i, p in enumerate(sent_probs)]
        ranked_sentiments.sort(key=lambda x: x[1], reverse=True)

        full_results.append({
            "timestamp": f"{start_t:.1f}s - {end_t:.1f}s",
            "transcript": transcript,
            "emotions": ranked_emotions,
            "sentiments": ranked_sentiments
        })

        if os.path.exists(clip_path):
            os.remove(clip_path)

    # --- Final Report ---
    print("\n" + "="*60)
    print(f"FINAL ANALYSIS REPORT: {os.path.basename(raw_video_path)}")
    print("="*60)

    for res in full_results:
        print(f"\nTimestamp: [{res['timestamp']}]")
        print(f"Transcript: \"{res['transcript']}\"")
        print("-" * 30)

        top_emo = res['emotions'][0]
        top_sent = res['sentiments'][0]
        print(f"Top Prediction: {top_emo[0].upper()} ({top_sent[0].upper()})")

        print("\nAll Emotions:")
        for emo, prob in res['emotions']:
            print(f"  {emo:<10} {prob:>6.1%} {'|' * int(prob * 20)}")

        print("\nAll Sentiments:")
        for sent, prob in res['sentiments']:
            print(f"  {sent:<10} {prob:>6.1%} {'|' * int(prob * 20)}")
        print("_" * 60)


if __name__ == "__main__":
    file_path = "video/test_video.py"
    analyze_video(raw_video_path=file_path)
