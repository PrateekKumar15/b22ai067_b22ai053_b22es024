import os
import torch
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2Config,
    Trainer, 
    TrainingArguments
)
from src.data.dataset import RobustSpeechDataset
from src.models.robust_wav2vec2 import RobustWav2Vec2ForCTC
from src.training.trainer import DataCollatorCTCWithPadding, get_tri_stage_scheduler

def main():
    # 1. Initialize Processor and Config
    pretrained_id = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(pretrained_id)
    vocab_size = len(processor.tokenizer)
    
    config = Wav2Vec2Config.from_pretrained(
        pretrained_id,
        vocab_size=vocab_size,
        # SpecAugment parameters (enforcing robustness against partial signal loss)
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.05,
        mask_feature_length=10
    )
    
    # 2. Initialize Model
    model = RobustWav2Vec2ForCTC(config, vocab_size=vocab_size)
    model.freeze_feature_extractor() # Optional: freeze CNN to save compute
    
    # 3. Setup Dataset
    # Provide dummy paths here - users must substitute with actual LibriSpeech and MS-SNSD directories
    train_dataset = RobustSpeechDataset(
        librispeech_paths=["dummy_path_1.wav", "dummy_path_2.wav"], 
        transcripts=["hello world", "test sequence"],
        noise_paths=["dummy_noise_1.wav"],
        snr_min=-5.0,
        snr_max=15.0
    )
    
    # 4. Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./wav2vec2-robust",
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=3e-5,    # Peak LR for Tri-stage scheduler
        weight_decay=0.01,     # AdamW weight decay (lambda = 0.01)
        save_total_limit=2,
    )
    
    # Custom Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    # Tri-stage config (e.g. 10% warmup, 40% hold, 50% decay)
    total_steps = 10000 
    scheduler = get_tri_stage_scheduler(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_hold_steps=int(0.4 * total_steps), 
        num_training_steps=total_steps
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset, # replace with actual validation set
        tokenizer=processor.feature_extractor,
        optimizers=(optimizer, scheduler)
    )
    
    print("Starting Training Pipeline...")
    trainer.train()

if __name__ == "__main__":
    main()
