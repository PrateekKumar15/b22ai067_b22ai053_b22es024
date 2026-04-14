# Noise-Robust Automatic Speech Recognition using Wav2Vec 2.0

*A framework bridging the gap between laboratory ASR metrics and real-world acoustic challenges.*

**Authors**:
- Manas Chechani (B22AI053)
- Prateek Kumar (B22ES024)
- Raj Vijayvargiya (B22AI067)
*Indian Institute of Technology Jodhpur | Speech and Audio Processing Course*

---

## 📌 Abstract & Motivation
State-of-the-Art (SOTA) ASR systems achieve near-human transcription on clean data but suffer catastrophic performance drops in highly reverberant or low-SNR conditions. Non-stationary noise (traffic, babble) destructively interferes with vocal formants, drastically increasing the Word Error Rate (WER). 

This project solves this by anchoring **dynamic noise injection** natively within a **multi-head cross-attention framework**, mapping noisy features straight to learned linguistic contexts using a fine-tuned HuggingFace Wav2Vec 2.0 Base Model.

---

## 📂 Project Structure

```text
Speech Project/
│
├── README.md                      # Project documentation
├── environment.yml                # Conda environment manifest
├── .gitignore                     # Git tracking exclusions
│
├── src/                           # Core PyTorch Modules
│   ├── data/
│   │   ├── dataset.py             # Dataloaders for LibriSpeech, MS-SNSD & Common Voice
│   │   └── preprocessor.py        # 16kHz resampling, SpecAugment, and Dynamic SNR mixing
│   │
│   ├── models/
│   │   └── robust_wav2vec2.py     # Custom RobustWav2Vec2ForCTC with Cross-Attention Filter
│   │
│   └── training/
│       └── trainer.py             # Tri-stage LR scheduler and CTC Padding Collator
│
├── test_run.py                    # Dry-run validation script for the model constraints
├── train.py                       # Main training hub initiating HuggingFace Trainer
├── evaluate.py                    # Fairness analytics & SNR Acoustic Stress Testing
└── plot_graphs.py                 # Generates spectrogram visuals and QR code assets
```

---

## 🚀 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/iit-jodhpur/noise-robust-wav2vec2.git
   cd noise-robust-wav2vec2
   ```

2. **Initialize the Environment**
   We have provided a fully reproducible YAML file for Conda.
   ```bash
   conda env create -f environment.yml
   conda activate asr_env
   ```
   *(Alternatively, if your environment is already configured, you can just install standard requirements via `pip install torch torchaudio transformers datasets`)*

---

## 💻 How to Run the Project

### 1. Validate the Architecture
Before downloading heavy datasets, ensure your environment matches our hardware optimizations.
```bash
python test_run.py 
```
*This simulates a dummy data run checking if the Cross-Attention and CTC structures compute loss mathematically accurately without crashing.*

### 2. Generate Visuals
If you want to view the difference between Clean Audio and Noisy Acoustic Degradation, run the visualization suite:
```bash
python plot_graphs.py
```
*Outputs `spectrogram.png` and `qrcode.png` directly into your main directory.*

### 3. Model Training
In `train.py`, setup the absolute paths pointing to your local `LibriSpeech` and `MS-SNSD` datasets.
```bash
python train.py
```
*Trains using the custom `DataCollatorCTCWithPadding` utilizing an AdamW optimizer ($\lambda=0.01$) governed by our Tri-stage learning rate scheduler peaking at $\eta=3 \times 10^{-5}$. Weight checkpoints are saved to the `wav2vec2-robust/` directory.*

### 4. Responsible AI Evaluation
Run comprehensive bias and fairness testing utilizing the Mozilla Common Voice set.
```bash
python evaluate.py
```
*Triggers our algorithmic evaluators that partition outputs structurally by demographic metadata validating `Demographic Parity` across genders and dialects, and computing exponential decay curves against critical SNR environments (Acoustic Stress-Testing).*
