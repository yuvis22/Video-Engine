import os
import glob
import subprocess
import shutil
import datetime
import imageio_ffmpeg
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig

def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def generate_srt(segments, srt_path):
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")

def process_video(input_file: str, output_dir: str):
    print(f"Processing: {input_file}")
    base_name = os.path.basename(input_file)
    name_no_ext, ext = os.path.splitext(base_name)
    
    # 1. Transcribe with LoRA Tuning
    print("Loading Base Whisper Model + Custom LoRA Adapters...")
    
    model_id = "openai/whisper-tiny"
    lora_path = os.path.join(os.path.dirname(__file__), "lora_whisper_output")
    
    # Load base model & processor
    processor = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Inject our locally trained LoRA weights!
    if os.path.exists(lora_path):
        print(f"✅ Found custom LoRA adapter at {lora_path}. Injecting weights...")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print(f"⚠️ LoRA adapter NOT found at {lora_path}. Falling back to standard Whisper...")
        model = base_model

    print("Extracting Audio & Transcribing...")
    
    # Use librosa to load audio directly from the video file format (ffmpeg backend)
    audio_array, sampling_rate = librosa.load(input_file, sr=16000)
    
    # Generate tokens
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            return_timestamps=True,
            language="hi",
            task="transcribe"
        )
    
    # Decode with timestamps
    transcription = processor.batch_decode(generated_tokens, decode_with_timestamps=True)[0]
    
    # We must manually parse the huggingface timestamp tags into segments
    # Huggingface outputs format like: "<|0.00|> Hello <|2.50|>"
    import re
    timestamp_pattern = re.compile(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", re.DOTALL)
    
    segments_list = []
    
    # If the model didn't use strict tag pairs, we will fallback to a single block
    matches = timestamp_pattern.findall(transcription)
    if matches:
        for match in matches:
            start_time, text, end_time = match
            segments_list.append({
                "start": float(start_time),
                "end": float(end_time),
                "text": text.strip()
            })
    else:
        # Fallback if timestamps fail
        clean_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        duration = len(audio_array) / 16000
        segments_list.append({"start": 0.0, "end": duration, "text": clean_text})

    print(f"Transcription complete. Generated {len(segments_list)} chunks.")

    # 2. Generate SRT
    srt_filename = f"{name_no_ext}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    generate_srt(segments_list, srt_path)
    print(f"SRT saved to: {srt_path}")

    # 3. Hardcode subtitles
    output_video_path = os.path.join(output_dir, f"{name_no_ext}_subtitled{ext}")
    print("Burning in subtitles with FFmpeg...")
    
    # Subtitle Style: Yellow text (&H0000FFFF in ASS format ABGR), BorderStyle=1 (Outline), Fontsize=16
    style = "FontSize=16,PrimaryColour=&H0000FFFF,BorderStyle=1"
    
    # Note: ffmpeg subtitles filter requires path escaping for absolute paths on some systems, 
    # but since we run it in the same dir or relative, we can just pass the path.
    # To be safe with colons in paths on windows, escaping is needed, but we are on macOS/Linux.
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", input_file,
            "-vf", f"subtitles={srt_path}:force_style='{style}'",
            "-c:a", "copy",
            output_video_path
        ]
        subprocess.run(cmd, check=True)
        print(f"Output video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")

def main():
    base_dir = os.path.dirname(__file__)
    input_queue = os.path.join(base_dir, "input_queue")
    output_results = os.path.join(base_dir, "output_results")
    
    os.makedirs(output_results, exist_ok=True)
    
    videos = glob.glob(os.path.join(input_queue, "*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {input_queue}.")
        return

    for video in videos:
        process_video(video, output_results)
        # Move processed video out of queue to avoid reprocessing
        processed_dir = os.path.join(input_queue, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        shutil.move(video, os.path.join(processed_dir, os.path.basename(video)))
        print(f"Moved {video} to processed folder.")

if __name__ == "__main__":
    main()
