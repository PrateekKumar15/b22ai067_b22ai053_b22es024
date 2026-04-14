import torch
from transformers import Wav2Vec2Processor
from src.models.robust_wav2vec2 import RobustWav2Vec2ForCTC
from src.data.dataset import CommonVoiceEvaluationDataset
from datasets import load_metric
import numpy as np
import pandas as pd

def evaluate_demographic_parity(model, processor, dataset):
    """
    Measures WER variance across distinct gender and dialect metadata.
    """
    wer_metric = load_metric("wer")
    
    results = []
    
    # Run Inference
    model.eval()
    for batch in dataset:
        input_values = batch["waveform"].unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            logits = model(input_values).get("logits")
            
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        ref_str = batch["transcript"]
        
        wer = wer_metric.compute(predictions=[pred_str], references=[ref_str])
        
        results.append({
            "gender": batch["metadata"].get("gender", "unknown"),
            "dialect": batch["metadata"].get("dialect", "unknown"),
            "wer": wer
        })
        
    df = pd.DataFrame(results)
    
    print("\n--- Demographic Parity Report ---")
    print("Aggregate WER by Gender:")
    print(df.groupby("gender")["wer"].mean())
    print("\nAggregate WER by Dialect:")
    print(df.groupby("dialect")["wer"].mean())
    

def acoustic_stress_test(model, processor, clean_waveform, ref_str, noise_waveform, snr_levels):
    """
    Maps performance decay curves at critical SNR thresholds.
    """
    from src.data.preprocessor import AudioPreprocessor
    wer_metric = load_metric("wer")
    
    print("\n--- Acoustic Stress Testing ---")
    model.eval()
    
    results = []
    for snr in snr_levels:
        preprocessor = AudioPreprocessor(snr_min=snr, snr_max=snr)
        noisy_wave = preprocessor(clean_waveform, 16000, noise_waveform, 16000)
        
        input_values = noisy_wave.unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model(input_values).get("logits")
            
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        
        wer = wer_metric.compute(predictions=[pred_str], references=[ref_str])
        print(f"SNR {snr}dB -> WER: {wer:.4f}")
        results.append({"snr": snr, "wer": wer})
        
    return pd.DataFrame(results)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Fine-tuned model mappings
    pretrained_id = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(pretrained_id)
    
    # For evaluation, load from your saved directory e.g., "./wav2vec2-robust"
    print("[NOTE] Attempting to load model - ensure trained weights exist.")
    # config = Wav2Vec2Config.from_pretrained("./wav2vec2-robust")
    # model = RobustWav2Vec2ForCTC.from_pretrained("./wav2vec2-robust", config=config)
    # model.to(device)
    
    print("Initialize evaluation pipelines here using valid test datasets...")
    # cv_dataset = CommonVoiceEvaluationDataset(...)
    # evaluate_demographic_parity(model, processor, cv_dataset)

if __name__ == "__main__":
    main()
