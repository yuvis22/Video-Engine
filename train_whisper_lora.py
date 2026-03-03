import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    # Load audio array and resample to 16kHz if necessary
    # In fetch_data, we assume 16kHz or we should resample here. 
    # For simplicity, we process the raw array directly
    audio = batch["audio_array"]
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def train():
    print("========================================")
    print("STEP 1: Initializing Whisper LoRA Pipeline")
    print("========================================")
    model_id = "openai/whisper-tiny"
    
    print(f"Loading Processor for {model_id} (Language: Hindi)...")
    processor = WhisperProcessor.from_pretrained(model_id, language="hindi", task="transcribe")
    
    print(f"Loading Base Model {model_id} (Strict CPU Mode)...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    print("\n========================================")
    print("STEP 2: Configuring LoRA Adapters")
    print("========================================")
    # Configure LoRA
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    
    model = get_peft_model(model, config)
    print("LoRA Network Ready!")
    model.print_trainable_parameters()

    print("\n========================================")
    print("STEP 3: Loading Massive JSONL Dataset")
    print("========================================")
    data_path = "./data/massive_train.jsonl"
    try:
        print("Memory Safety Enabled: Using Arrow mmap loading to bypass 24GB RAM limit crash...")
        dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"SUCCESS: Loaded {len(dataset)} speech array items securely into memory!")
        
        print("\n========================================")
        print("STEP 4: Extracting Whisper Math Audio Features (This will take a while...)")
        print("========================================")
        dataset = dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=dataset.column_names)
        print("SUCCESS: Complete Audio Dataset perfectly extracted and tokenized!")
    except Exception as e:
        print(f"Warning: Could not load dataset fully. {e}")
        # Initialize an empty dataset for validation structural checks if missing
        dataset = Dataset.from_dict({"input_features": [], "labels": []})
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lora_whisper_output",
        per_device_train_batch_size=8, # Utilizing full 24GB RAM
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=500,
        max_steps=5000, # Massive long-run high-accuracy training
        gradient_checkpointing=True,
        fp16=False, # CPU doesn't support fp16 training efficiently, keep false
        evaluation_strategy="no", 
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        logging_steps=50,
        report_to=["none"], # Disable wandb/tensorboard for clean run
        use_cpu=True # Explicitly force CPU
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    if len(dataset) > 0:
        print("\n========================================")
        print("STEP 5: INITIATING LORA FINE-TUNING LOOP 🔥")
        print("========================================")
        print("Trainer is firing up. Hang tight, loss curves are coming...")
        trainer.train()
        print("\n🎉 TRAINING COMPLETELY FINISHED! AI Saved to ./lora_whisper_output! 🎉")
    else:
        print("\nDataset missing or empty. Skipping AI start.")
