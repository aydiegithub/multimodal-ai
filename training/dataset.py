from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
import os
from colorama import Fore
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
from typing import Dict, Any, Optional

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LEN_OF_FRAMES: int = 30
TIME_STEPS: int = 300


class MELDDataset(Dataset):
    def __init__(self, csv_path: str, video_dir: str, tokenizer_name: str = "bert-base-uncased") -> None:
        self.data: pd.DataFrame = pd.read_csv(csv_path)
        self.video_dir: str = video_dir
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name)

        # Emotion and sentiment mappings
        self.emotion_map: Dict[str, int] = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }
        self.sentiment_map: Dict[str, int] = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

    def _load_video_frames(self, video_path: str) -> torch.FloatTensor:
        cap = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Cannot read first frame: {video_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < LEN_OF_FRAMES and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video Error: {e}") from e
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from {video_path}")

        if len(frames) < LEN_OF_FRAMES:
            padding_count = LEN_OF_FRAMES - len(frames)
            frames += [np.zeros_like(frames[0])] * padding_count
        else:
            frames = frames[:LEN_OF_FRAMES]

        tensor_frames = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
        return tensor_frames

    def _extract_audio_features(self, video_path: str) -> torch.Tensor:
        audio_path: str = video_path.replace(".mp4", ".wav")
        try:
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path,
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec: torch.Tensor = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < TIME_STEPS:
                padding = TIME_STEPS - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :TIME_STEPS]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {e}") from e
        except Exception as e:
            raise ValueError(f"Audio error: {e}") from e
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        if isinstance(index, torch.Tensor):
            index = index.item()
        row = self.data.iloc[index]

        video_filename: str = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path: str = os.path.join(self.video_dir, video_filename)

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No video found: {path}")

            print(Fore.GREEN +
                  f"[INFO] File found: {video_filename}" + Fore.RESET)

            text_inputs = self.tokenizer(
                row["Utterance"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            emotion_label: torch.Tensor = torch.tensor(
                self.emotion_map[row['Emotion'].lower()], dtype=torch.long)
            sentiment_label: torch.Tensor = torch.tensor(
                self.sentiment_map[row['Sentiment'].lower()], dtype=torch.long)

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze(),
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': emotion_label,
                'sentiment_label': sentiment_label
            }

        except Exception as e:
            print(
                Fore.RED + f"[ERROR] Processing {video_filename}: {e}" + Fore.RESET)
            raise


def collate_fn(batch):
    # filter out none samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv: str, train_video_dir: str,
                        dev_csv: str, dev_video_dir: str,
                        test_csv: str, test_video_dir: str,
                        batch_size: int = 32):

    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        train_csv="../dataset/train/train_sent_emo.csv",
        train_video_dir="../dataset/train/train_splits",

        dev_csv="../dataset/dev/dev_sent_emo.csv",
        dev_video_dir="../dataset/dev/dev_splits_complete",

        test_csv="../dataset/test/test_sent_emo.csv",
        test_video_dir="../dataset/test/output_repeated_splits_test",

        batch_size=32
    )

    dataset = MELDDataset(
        csv_path="../dataset/dev/dev_sent_emo.csv",
        video_dir="../dataset/dev/dev_splits_complete"
    )

    for batch in train_loader:
        print("\n" + "*" * 60)
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])

        break
