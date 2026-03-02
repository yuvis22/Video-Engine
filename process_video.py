import os
import glob
import subprocess
import shutil
from faster_whisper import WhisperModel
import datetime
import imageio_ffmpeg

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
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")

def process_video(input_file: str, output_dir: str):
    print(f"Processing: {input_file}")
    base_name = os.path.basename(input_file)
    name_no_ext, ext = os.path.splitext(base_name)
    
    # 1. Transcribe
    print("Loading faster-whisper model (small, int8)...")
    # Using cpu and int8 as specified
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print("Transcribing...")
    # Target: English-Hindi code-switching. We can leave language unset for auto-detection or set to None
    segments, info = model.transcribe(input_file, beam_size=5)
    
    # Needs to be consumed to get segments
    segments_list = list(segments)
    print(f"Transcription complete. Detected language: {info.language} with probability {info.language_probability:.2f}")

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
