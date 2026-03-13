# Physics-Inspired Underwater Acoustic Target Recognition (UATR)

This repository contains the official PyTorch implementation for our proposed **HTAN (Harmonic-Temporal Attention Network)**. 

Designed specifically for the challenging underwater acoustic environment, HTAN achieves high-accuracy target recognition through a highly compact, physics-inspired architecture. By seamlessly integrating acoustic physical priors into a data-driven framework, HTAN completely discards the reliance on parameter-heavy, generic pre-trained models (e.g., Audio Spectrogram Transformers).

## 🚀 Key Innovations

1. **Multi-Scale Contextual Frontend:** Parallel local, spectral, and temporal convolutional branches effectively capture both transient cavitation impulses (broadband) and continuous mechanical line spectra (narrowband).
2. **Harmonic Frequency GCN:** We introduce a *physics-inspired frequency prior* to mask the dynamic graph attention. This strictly constrains the network to learn genuine harmonic resonance rather than overfitting to environmental background noise.
3. **Temporal Attention Pooling:** A 1-layer BiGRU paired with a frame-level attention mechanism dynamically aggregates the most informative acoustic events across the time dimension, ignoring noise-only frames.
4. **Edge-Deployment Friendly:** Trained entirely from scratch, the model possesses fewer than 1.5 million parameters, making it highly suitable for bandwidth-constrained and power-limited Underwater Acoustic Sensor Networks (UASNs).

## 📂 Repository Structure

```text
.
├── Datasets/
│   ├── ShipsEar_Data_Preprocessing.py  # 16kHz resampling & strict 5s segmentation
│   └── ShipsEar_dataloader.py          # PyTorch Lightning DataModule
├── src/
│   └── models/
│       └── custom_model.py             # Core implementation of HTAN (Network Architecture)
├── Utils/
│   ├── LitModel.py                     # LightningModule wrapper (Train/Val/Test logic & Metrics)
│   └── LogMelFilterBank.py             # Acoustic Feature Extraction (Log Mel Spectrogram)
├── demo_light.py                       # Main training and evaluation script
├── Demo_Parameters.py                  # Hyperparameter configurations
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## ⚙️ Installation & Environment Setup

Ensure you have Python 3.8+ installed. We highly recommend using a virtual environment (e.g., Conda). Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Ensure `torch`, `lightning`, `librosa`, and `torchmetrics` are properly installed and compatible with your CUDA version for GPU acceleration).*

## 📊 Dataset Preparation

We utilize the widely recognized public **ShipsEar** dataset. 

1. Download the raw audio files from the official ShipsEar repository.
2. Place the audio files into folders `A`, `B`, `C`, `D`, and `E` corresponding to their respective vessel classes inside a root directory named `shipsEar_AUDIOS/`.
3. The training script will automatically trigger `Generate_Segments` to resample the audio to 16kHz and slice it into strict 5-second non-overlapping segments to prevent data leakage.

## 🏃‍♂️ How to Run

To train the HTAN model from scratch, execute the main script. 
Since HTAN is a lightweight model trained from scratch, we recommend using a larger batch size (`64`) and an initial learning rate of `1e-3`.

```bash
python demo_light.py \
    --model HTAN \
    --data_selection 1 \
    --train_batch_size 64 \
    --lr 1e-3 \
    --num_epochs 100 \
    --audio_feature LogMelFBank \
    --number_mels 128
```

### Monitoring Training
This project integrates PyTorch Lightning's TensorBoard logger. You can monitor the training progress, validation accuracy (`val_acc`), and Area Under the Precision-Recall Curve (`val_auprc`) in real-time by running:

```bash
tensorboard --logdir=tb_logs/
```

## 🔬 Ablation Studies (For Paper Validation)

To validate the acoustic physical assumptions proposed in our paper, you can easily conduct ablation studies by modifying `src/models/custom_model.py`:

- **w/o Physics Prior (Pure Data-Driven Graph):** In `HarmonicFrequencyGCN.forward()`, comment out the prior masking step: `A_logits = A_logits.masked_fill(~prior_mask, -1e9)`.
- **w/o Temporal Evolution:** Replace the `BiGRU` temporal encoder with a simple temporal mean pooling operation.
- **w/o Multi-Scale Frontend:** Remove `branch2` (1x7) and `branch3` (7x1) in the `MultiScaleConvBlock`, reverting it to a standard local CNN.
- **w/o Context Branch:** Comment out `branch4_pool` to evaluate the impact of the lightweight global SE-style context bias.

---
