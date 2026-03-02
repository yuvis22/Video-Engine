import json
import os
from datasets import load_dataset, Audio

def fetch_datasets(output_file: str, num_samples: int = 50000):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset configurations for streaming
    # Using simple text datasets to bypass audio decoding dependencies on this system.
    # We will generate synthetic code-mixed proxy data for the pipeline input demonstration.
    # WMT16 or similar translation datasets provide clean text pairs. Let's use `opus_wikipedia` or just take English and Hindi from a common text corpus.
    # Let's use the 'allenai/c4' dataset for English and 'ai4bharat/IndicParaphrase' or similar simple text for Hindi.
    # The simplest working streamable text datasets with hi and en splits are wikipedia extracts or cc100.
    # Massive accurate datasets for high efficiency training
    dataset_configs = [
        ("librispeech_asr", "clean", "train.360"), # 360 hours of extremely clean English
        ("google/xtreme_s", "minds14.hi-IN", "train"), # Clean Hindi speech
    ]

    samples_collected = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for repo_id, config_name, split in dataset_configs:
            if samples_collected >= num_samples:
                break
                
            print(f"Downloading {repo_id} ({config_name} - {split}) locally to avoid stream thread crash...")
            try:
                # `trust_remote_code=True` removed because newer Hugging Face `datasets` versions completely obsolete it
                # and throw hard errors if passed.
                ds = load_dataset(repo_id, config_name, split=split)
                
                # Resample cleanly to 16kHz (Whisper standard)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                
                samples_to_take = num_samples // len(dataset_configs)
                
                for i, sample in enumerate(ds):
                    if i >= samples_to_take or samples_collected >= num_samples:
                        break
                    
                    # Extract the text
                    text = sample.get('transcription') or sample.get('sentence') or sample.get('text')
                    
                    # Store audio array directly as a list to save in JSONL
                    audio_data = sample.get('audio', {})
                    audio_array = audio_data.get('array')
                    sampling_rate = audio_data.get('sampling_rate')
                    
                    if text and audio_array is not None:
                        record = {
                            "dataset": repo_id,
                            "language": config_name,
                            "text": text,
                            "audio_array": audio_array.tolist() if hasattr(audio_array, 'tolist') else list(audio_array),
                            "sampling_rate": sampling_rate
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        samples_collected += 1
            except Exception as e:
                print(f"Error loading {repo_id} ({config_name}): {e}")

    print(f"Data fetching complete. Saved {samples_collected} samples to {output_file}.")

if __name__ == "__main__":
    # Save the huge target text/audio array JSONL to data folder
    output_path = os.path.join(os.path.dirname(__file__), "data", "massive_train.jsonl")
    fetch_datasets(output_path, 50000)
