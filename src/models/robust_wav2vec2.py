import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, PreTrainedModel, Wav2Vec2Config
import math

class CrossAttentionFilter(nn.Module):
    """
    Multi-Head Cross-Attention Filtering
    Mathematical Formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Q: Noisy features from Temporal CNN
    K, V: Learned linguistic context (or self-attended representations)
    """
    def __init__(self, d_model=768, nhead=8, num_context_vectors=128):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Learned linguistic context vectors
        self.context_embeddings = nn.Parameter(torch.randn(1, num_context_vectors, d_model))
        
        self.mhca = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, noisy_features):
        """
        noisy_features: Output from Wav2Vec2 feature_extractor (CNN)
        Shape: (batch, seq_len, d_model)
        """
        B = noisy_features.size(0)
        
        # Q = noisy_features, K = V = context_embeddings
        Q = noisy_features
        K = self.context_embeddings.expand(B, -1, -1)
        V = self.context_embeddings.expand(B, -1, -1)
        
        filtered_features, _ = self.mhca(Q, K, V)
        
        # Residual connection and LayerNorm
        out = self.layer_norm(noisy_features + self.dropout(filtered_features))
        return out

class RobustWav2Vec2ForCTC(nn.Module):
    """
    Noise-Robust Automatic Speech Recognition using Wav2Vec 2.0
    Implements:
    1. Feature Encoder (7-Layer Temporal CNN)
    2. Cross-Attention Filter
    3. Transformer Context Network
    4. CTC Decoder
    """
    def __init__(self, config: Wav2Vec2Config, vocab_size: int):
        super().__init__()
        self.config = config
        
        # Base Wav2Vec2 Model (contains CNN feature extractor + Transformer encoder)
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # Custom Cross Attention Filter inserted after Feature Extractor
        self.cross_attention = CrossAttentionFilter(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads
        )
        
        # Linear projection to vocabulary for CTC Loss
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
    
    def freeze_feature_extractor(self):
        """Freezes the 7-Layer Temporal CNN"""
        self.wav2vec2.feature_extractor._freeze_parameters()
        
    def forward(self, input_values, attention_mask=None, labels=None):
        """
        Forward pass.
        We dissect the Wav2Vec2 forward pass to insert our Cross-Attention Filter.
        """
        # 1. Self-Supervised Feature Encoder (7-Layer CNN)
        extract_features = self.wav2vec2.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2) # (batch, seq_len, hidden_size)
        
        # 2. Add Positional Embeddings & feature projection (as per standard Wav2Vec2)
        hidden_states, extract_features = self.wav2vec2.feature_projection(extract_features)
        
        # 3. Transformer Context Network (Multi-Head Cross-Attention Filtering)
        # Apply our custom mathematical formulation for filtering noise
        filtered_hidden_states = self.cross_attention(hidden_states)
        
        # Continue with standard Wav2Vec2 Transformer Encoder
        encoder_outputs = self.wav2vec2.encoder(
            filtered_hidden_states,
            attention_mask=attention_mask
        )
        
        sequence_output = encoder_outputs[0]
        
        # 4. Linear Projection & CTC Decoder
        sequence_output = self.dropout(sequence_output)
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            # Objective Function: CTC Loss
            # L_CTC = -log P(Y|X)
            
            # Retrieve lengths
            input_lengths = self._get_feat_extract_output_lengths(input_values.shape[1])
            if attention_mask is not None:
                # compute real output lengths according to attention mask
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            else:
                input_lengths = torch.full((input_values.shape[0],), input_lengths, dtype=torch.long, device=logits.device)
                
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels[labels_mask]
            
            # ctc_loss in PyTorch expects (T, N, C)
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_fn = nn.CTCLoss(blank=self.config.pad_token_id, reduction="mean", zero_infinity=True)
                loss = loss_fn(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths
                )

        return {"loss": loss, "logits": logits}

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the sequence length after CNN convolutions
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
