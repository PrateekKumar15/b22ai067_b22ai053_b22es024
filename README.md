# Noise-Robust ASR with Wav2Vec2

This project builds a noise-robust automatic speech recognition (ASR) pipeline on top of Wav2Vec2, with custom data augmentation and a custom model head designed to improve transcription quality in noisy environments.

Authors:

- Manas Chechani (B22AI053)
- Prateek Kumar (B22ES024)
- Raj Vijayvargiya (B22AI067)

Institute:

- Indian Institute of Technology Jodhpur

## 1) What This Project Does (Complete Description)

Modern ASR systems often perform very well on clean speech but degrade significantly under real-world conditions such as traffic noise, crowd noise, and low signal-to-noise ratio (SNR). This project focuses on making ASR more robust by combining:

- Dynamic noise mixing during data preparation
- A custom cross-attention filtering block inside a Wav2Vec2-style model
- CTC-based training for transcription
- Evaluation utilities for demographic parity and acoustic stress testing
- Visual and demo assets for reporting and presentation

In practical terms, the full project aims to answer:

- Can we improve transcription stability when audio quality drops?
- How does model performance change as SNR decreases?
- Does error vary across metadata groups such as gender or dialect/accent?

## 2) End-to-End Pipeline

The project pipeline is:

1. Data preparation and augmentation
2. Model construction and training
3. Evaluation under clean and noisy settings
4. Visualization and demo

### 2.1 Data Preparation

- Input speech can come from clean datasets (for example LibriSpeech).
- Noise samples can come from datasets such as MS-SNSD.
- The preprocessing pipeline in `src/data/preprocessor.py`:
  - resamples to 16 kHz
  - converts to mono if needed
  - mixes noise at random SNR between configurable bounds
  - normalizes mixed output to avoid clipping

Noise scaling follows standard SNR control:

$$
  SNR_{dB} = 10 \log_{10}(P_{signal} / P_{noise})
$$

### 2.2 Model

The core model in `src/models/robust_wav2vec2.py` uses:

- Wav2Vec2 feature extractor and encoder components
- A custom `CrossAttentionFilter`
- Dropout + linear projection head for token logits
- CTC loss for alignment-free sequence transcription

High-level flow:

1. Raw audio -> convolutional feature extractor
2. Feature projection -> hidden states
3. Cross-attention filtering of hidden states
4. Transformer encoder processing
5. Linear LM head -> logits
6. CTC loss when labels are provided

### 2.3 Training Strategy

Training utilities in `src/training/trainer.py` provide:

- `DataCollatorCTCWithPadding` for dynamic batch padding
- Tri-stage learning-rate scheduler:
  - Warmup (linear increase)
  - Hold (constant)
  - Exponential decay

### 2.4 Evaluation and Analysis

`evaluate.py` contains skeletons/utilities for:

- Demographic parity evaluation (grouped WER statistics)
- Acoustic stress testing over multiple SNR levels

This is intended to quantify robustness and fairness behavior, not just average clean-set WER.

## 3) Repository Structure

```text
Speech Project/
|- README.md
|- environment.yml
|- Speech project master code.ipynb
|- train.py
|- evaluate.py
|- test_run.py
|- plot_graphs.py
|- src/
|  |- data/
|  |  |- dataset.py
|  |  |- preprocessor.py
|  |- models/
|  |  |- robust_wav2vec2.py
|  |- training/
|     |- trainer.py
```

File purpose summary:

- `train.py`: baseline training entry point using Hugging Face `Trainer`
- `evaluate.py`: evaluation utility scaffold for fairness + SNR stress testing
- `test_run.py`: fast architecture sanity test (forward pass + CTC loss)
- `plot_graphs.py`: utility to create spectrogram and QR code assets
- `Speech project master code.ipynb`: complete Colab-oriented workflow (data, patching, training, evaluation, demo)

## 4) Environment Setup (Local)

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate asr_env
```

### Option B: pip + venv

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows PowerShell

pip install torch torchvision torchaudio transformers datasets librosa soundfile numpy pandas wandb evaluate jiwer tqdm qrcode
```

## 5) How To Run Locally

### 5.1 Quick sanity check

```bash
python test_run.py
```

Expected:

- model forward pass runs
- loss/logits exist
- no crash in custom architecture path

### 5.2 Generate visuals

```bash
python plot_graphs.py
```

Expected outputs in project root:

- `spectrogram.png`
- `qrcode.png`

### 5.3 Training

```bash
python train.py
```

Important note:

- Current `train.py` uses placeholder dummy paths and dummy transcripts by default.
- Before real training, replace dummy paths with real dataset paths or connect it to a Hugging Face dataset flow.

Training outputs:

- checkpoints and training artifacts in `./wav2vec2-robust`

### 5.4 Evaluation

```bash
python evaluate.py
```

Important note:

- `evaluate.py` currently contains scaffold logic and commented model-loading blocks.
- You must provide real trained checkpoint paths and real evaluation dataset objects before running full evaluation.

## 6) Running on Cloud (Google Colab) Using `Speech project master code.ipynb`

This is the most complete runnable workflow in the repository for cloud execution.

The notebook includes:

- dependency installation
- repository cloning inside Colab
- auto-patching of project files for compatibility
- streamed LibriSpeech loading from Hugging Face
- MS-SNSD noise download
- noisy test-set creation at multiple SNR levels
- model training
- comparison against vanilla Wav2Vec2 and Whisper-base
- stress-test plots
- optional Gradio demo

### 6.1 Open notebook in Colab

1. Open Google Colab.
2. Set runtime: GPU (T4 or better).
3. Open `Speech project master code.ipynb`.

Optional direct GitHub URL pattern:

```text
https://colab.research.google.com/github/<owner>/<repo>/blob/main/Speech%20project%20master%20code.ipynb
```

### 6.2 Run cells in order

The notebook is structured in four practical blocks:

1. Data preparation notebook block
2. Training notebook block
3. Evaluation and SOTA comparison block
4. Interactive Gradio demo block

Do not skip order. Later blocks depend on files generated earlier.

### 6.3 What each notebook block produces

Block 1 (data):

- creates `/content/asr_data`
- streams LibriSpeech samples
- clones MS-SNSD and prepares noise files
- builds noisy manifests and saves clean/noisy visualizations

Block 2 (training):

- initializes robust model and processor
- creates datasets/collator/scheduler
- trains model
- saves model artifacts to `/content/wav2vec2_robust_final`
- saves training plots and zipped model

Block 3 (evaluation):

- loads robust model, vanilla Wav2Vec2, and Whisper-base
- evaluates WER at Clean, 15 dB, 5 dB, 0 dB
- generates comparison tables and plots
- saves results to `/content/results`

Block 4 (demo):

- launches Gradio UI for live audio transcription comparison
- lets you test clean/noisy conditions interactively

### 6.4 Colab runtime tips

- Keep runtime on GPU throughout training/evaluation.
- If Colab disconnects, re-run setup cells and restore saved artifacts.
- Download key artifacts after each major phase:
  - model zip
  - results zip
  - generated plots

### 6.5 Important compatibility note

The notebook patches some repository source files at runtime to handle library-version differences and training/evaluation edge cases. This is intentional in the notebook workflow.

If your local scripts and notebook behavior differ, treat the notebook as the cloud reference pipeline.

## 7) Expected Outputs

Depending on how much of the pipeline you run, you should obtain:

- trained checkpoints for robust Wav2Vec2
- training curves (loss/WER)
- WER comparison across noise levels
- acoustic stress-test curve
- spectrogram visualizations
- optional interactive Gradio demo link

## 8) Current Limitations

- `train.py` in repo root is a minimal baseline with dummy dataset paths by default.
- `evaluate.py` is a scaffold and requires real dataset wiring before full use.
- Full end-to-end reproducibility in cloud is currently best represented by `Speech project master code.ipynb`.

## 9) Recommended Usage Path

If your goal is to run the complete project now:

1. Use Google Colab with `Speech project master code.ipynb`.
2. Run all cells sequentially.
3. Download final model and result artifacts.

If your goal is codebase cleanup and productionization:

1. Port notebook patches back into source files in `src/`.
2. Replace dummy dataset paths in `train.py` with real data loaders.
3. Complete model-loading and dataset hooks in `evaluate.py`.
