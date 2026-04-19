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

## 💻 How to Run the Project (In Depth Guide)

This section provides a detailed step-by-step guide to executing each script in the pipeline, explaining what happens under the hood and what inputs/outputs to expect.

### 1. Validate the Architecture (`test_run.py`)
**Purpose**: Before downloading heavy datasets or initiating long training loops, this script ensures your environment supports the architectural modifications.
- **Action**: Run the command below to perform a fast forward-pass dummy check.
  ```bash
  python test_run.py 
  ```
- **What it does**: It simulates a mini-batch with random tensors mirroring realistic speech features. It verifies that the `RobustWav2Vec2ForCTC` base class (with cross-attention modifications) correctly computes CTC Loss.
- **Expected Output**: A terminal confirmation that the loss computed successfully, ensuring no memory leaks or dimension mismatches exist in the custom pipeline.

### 2. Generate Visual Assets (`plot_graphs.py`)
**Purpose**: Generates visual comparisons of acoustic features under degrading Signal-to-Noise Ratio (SNR) environments, as well as necessary project QR assets.
- **Action**: Execute the visualization script.
  ```bash
  python plot_graphs.py
  ```
- **What it does**: 
  - Uses `matplotlib` to generate and save `spectrogram.png`, which plots a side-by-side Mel-Spectrogram comparison. On the left, it plots clean acoustic features, and on the right, it simulates acoustic degradation by mixing in Gaussian noise (low SNR).
  - Uses `qrcode` to generate `qrcode.png` containing a link to this GitHub project for posters or presentations.
- **Expected Output**: Two image files (`spectrogram.png`, `qrcode.png`) saved directly into your root directory.

### 3. Model Training Pipeline (`train.py`)
**Purpose**: Complete end-to-end training procedure integrating our custom HuggingFace datasets and `Trainer`.
- **Preparation needed**: Open `train.py` and modify the paths in `RobustSpeechDataset` (e.g. `librispeech_paths` and `noise_paths`) to match the absolute directories of your LibriSpeech corpus and MS-SNSD background noise collection.
- **Action**: Start model training.
  ```bash
  python train.py
  ```
- **What it does**: 
  - Initializes `facebook/wav2vec2-base` from HuggingFace, wrapped with our custom `RobustWav2Vec2ForCTC`. The core CNN feature extractor is frozen to save computational overhead.
  - SpecAugment parameters (Time and Feature masking) are rigorously enforced to build robustness against partial signal distortion.
  - Data batches are properly structured using `DataCollatorCTCWithPadding`.
  - Utilizes an `AdamW` optimizer ($\lambda=0.01$) alongside a Tri-stage learning rate scheduler (warmup, hold, exponential decay) peaking at $\eta=3 \times 10^{-5}$.
- **Expected Output**: Granular training logs printing to the terminal at `logging_steps=100`. Resulting model weights and optimizer progress checkpoints are saved (every 500 steps) directly to the `./wav2vec2-robust/` directory.

### 4. Responsible AI Evaluation (`evaluate.py`)
**Purpose**: Evaluates your fine-tuned model against ethical bias metrics and absolute acoustic stress tests to validate real-world readiness.
- **Preparation needed**: Ensure you have model weights stored inside `./wav2vec2-robust`. You must uncomment the loading logic in `evaluate.py` and define your data stream for the Mozilla Common Voice Evaluation Dataset.
- **Action**: Trigger the integrated evaluators.
  ```bash
  python evaluate.py
  ```
- **What it does**:
  - **Demographic Parity Tracking**: Evaluates the model sequentially, computing the Word Error Rate (WER). Iterates via metadata grouping to isolate WER by precise demographics (e.g. `gender`, `dialect`) through the `evaluate_demographic_parity` function utilizing pandas mapping.
  - **Acoustic Stress Testing**: Synthetically maps acoustic environments at various low-SNR thresholds with `acoustic_stress_test` to detect the exact drop-off points where WER decays exponentially.
- **Expected Output**: Generates two detailed summary tables directly out to your console: a Demographic Parity Report indicating exact acoustic biases/WER across genders and dialects, mapping alongside the algorithmic Acoustic Stress Report.
