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
    print("STEP 3: Streaming Massive Dataset Live from HuggingFace")
    print("========================================")
    
    try:
        print("Storage Bypass Enabled: Streaming dataset live to prevent 200GB disk crash...")
        
        # Stream the dataset directly from HF using a natively streamable dataset
        dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0", 
            "hi", 
            split="train", 
            streaming=True, 
            trust_remote_code=True
        )
        
        # Use datasets.Audio to cast on the fly
        from datasets import Audio
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        def prepare_dataset_stream(batch):
            audio = batch["audio"]["array"]
            batch["input_features"] = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
            batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
            return batch

        print("\n========================================")
        print("STEP 4: Extracting Whisper Math Audio Features On-The-Fly")
        print("========================================")
        dataset = dataset.map(prepare_dataset_stream)
        print("SUCCESS: Live streaming pipeline firmly established!")

    except Exception as e:
        print(f"Warning: Could not stream dataset. {e}")
        return
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lora_whisper_output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=500,
        max_steps=5000, 
        gradient_checkpointing=True,
        fp16=False,
        eval_strategy="no", 
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        logging_steps=50,
        report_to=["none"],
        use_cpu=True
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n========================================")
    print("STEP 5: INITIATING LORA FINE-TUNING LOOP 🔥")
    print("========================================")
    print("Trainer is firing up. Hang tight, loss curves are coming...")
    trainer.train()
    print("\n🎉 TRAINING COMPLETELY FINISHED! AI Saved to ./lora_whisper_output! 🎉")

if __name__ == "__main__":
    train()
