import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2Processor
from torch.optim.lr_scheduler import LambdaLR

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: int = None
    pad_to_multiple_of: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["waveform"]} for feature in features]
        label_features = [{"input_ids": self.processor.tokenizer(feature["transcript"]).input_ids} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.tokenizer.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def get_tri_stage_scheduler(optimizer, num_warmup_steps, num_hold_steps, num_training_steps):
    """
    Tri-stage learning rate scheduler:
    1. Warm-up (Linear increase from 0 to max_lr)
    2. Hold (Constant at max_lr)
    3. Exponential decay
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_hold_steps:
            return 1.0
        else:
            decay_steps = num_training_steps - num_warmup_steps - num_hold_steps
            current_decay_step = current_step - num_warmup_steps - num_hold_steps
            # Exponential decay
            return math.exp(-5.0 * current_decay_step / max(1, decay_steps))

    import math
    return LambdaLR(optimizer, lr_lambda)
