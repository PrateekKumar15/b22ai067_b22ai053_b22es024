import torch
from transformers import Wav2Vec2Config
from src.models.robust_wav2vec2 import RobustWav2Vec2ForCTC

def test_model():
    print("Testing Model Architecture & Forward Pass...")
    dummy_vocab_size = 32
    
    # Tiny configuration to prevent OOM and speed up test
    config = Wav2Vec2Config(
        vocab_size=dummy_vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3)
    )
    
    model = RobustWav2Vec2ForCTC(config, vocab_size=dummy_vocab_size)
    
    # Dummy input: batch_size=2, audio_length=8000
    dummy_input = torch.randn(2, 8000)
    
    # Dummy labels: batch_size=2, arbitrary length
    dummy_labels = torch.randint(0, dummy_vocab_size, (2, 50))
    
    output = model(input_values=dummy_input, labels=dummy_labels)
    
    assert "loss" in output, "Loss not returned by model."
    assert "logits" in output, "Logits not returned by model."
    print("SUCCESS: Forward pass correctly computes CTC Loss without crashing.")
    print("Calculated Loss:", output["loss"].item())
    print("Logits Shape:", output["logits"].shape)

if __name__ == "__main__":
    test_model()
