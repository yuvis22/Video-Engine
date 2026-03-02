import json
import os
from datasets import load_dataset

def fetch_datasets(output_file: str, num_samples: int = 500):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset configurations for streaming
    # Using simple text datasets to bypass audio decoding dependencies on this system.
    # We will generate synthetic code-mixed proxy data for the pipeline input demonstration.
    # WMT16 or similar translation datasets provide clean text pairs. Let's use `opus_wikipedia` or just take English and Hindi from a common text corpus.
    # Let's use the 'allenai/c4' dataset for English and 'ai4bharat/IndicParaphrase' or similar simple text for Hindi.
    # The simplest working streamable text datasets with hi and en splits are wikipedia extracts or cc100.
    # Dataset configurations for OCI deployment
    # Using text-only dataset to bypass `torchcodec` and PyGILState Linux audio decoding crash
    dataset_configs = [
        ("tatoeba", "en-hi", "train"),
    ]

    samples_collected = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for repo_id, config_name, split in dataset_configs:
            if samples_collected >= num_samples:
                break
                
            print(f"Streaming from {repo_id} ({config_name} - {split})...")
            try:
                # `trust_remote_code=True` required for some gated/script-based datasets
                ds = load_dataset(repo_id, config_name, split=split, streaming=True, trust_remote_code=True)
                
                samples_to_take = num_samples // len(dataset_configs)
                
                for i, sample in enumerate(ds):
                    if i >= samples_to_take or samples_collected >= num_samples:
                        break
                    
                    # Extract the text
                    translation = sample.get('translation', {})
                    text_en = translation.get('en')
                    text_hi = translation.get('hi')
                    
                    if text_en and text_hi:
                        # Write EN
                        record_en = {
                            "dataset": repo_id,
                            "language": "en",
                            "text": text_en,
                            "audio_array": [], # Mock audio array so train script doesn't crash
                            "sampling_rate": 16000
                        }
                        f.write(json.dumps(record_en, ensure_ascii=False) + "\n")
                        samples_collected += 1
                        
                        if samples_collected < num_samples:
                            # Write HI
                            record_hi = {
                                "dataset": repo_id,
                                "language": "hi",
                                "text": text_hi,
                                "audio_array": [],
                                "sampling_rate": 16000
                            }
                            f.write(json.dumps(record_hi, ensure_ascii=False) + "\n")
                            samples_collected += 1
            except Exception as e:
                print(f"Error loading {repo_id} ({config_name}): {e}")

    print(f"Data fetching complete. Saved {samples_collected} samples to {output_file}.")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "data", "mini_train.jsonl")
    fetch_datasets(output_path, 500)
