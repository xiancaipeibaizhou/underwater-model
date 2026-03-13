# Physics-Inspired Underwater Acoustic Target Recognition (UATR)

This repository contains the official PyTorch implementation for our proposed **HTAN (Harmonic-Temporal Attention Network)**.

Designed specifically for the challenging underwater acoustic environment, HTAN presents a compact and physics-inspired framework for underwater acoustic target recognition. By integrating acoustic physical priors into a data-driven framework, HTAN aims to provide an effective recognition approach that is trained entirely from scratch, without relying on parameter-heavy generic pre-trained backbones.

## 🚀 Key Innovations

1. **Multi-Scale Contextual Frontend:** Parallel local, spectral, and temporal convolutional branches are designed to capture both transient cavitation impulses (broadband) and continuous mechanical line spectra (narrowband).
2. **Harmonic Frequency GCN:** We introduce a *physics-inspired frequency prior* to mask the dynamic graph attention. This prior guides the network to emphasize harmonic-consistent frequency relations and suppress implausible cross-band interactions, rather than relying solely on data-driven attention.
3. **Temporal Attention Pooling:** A 1-layer BiGRU paired with a frame-level attention mechanism aggregates informative acoustic events across the time dimension.
4. **Compact Architecture:** Trained entirely from scratch, the model possesses fewer than 1.5 million parameters. Its compact parameterization makes it a promising candidate for resource-constrained deployment scenarios, such as Underwater Acoustic Sensor Networks (UASNs).

## 📂 Repository Structure

    .
    ├── Datasets/
    │   ├── ShipsEar_Data_Preprocessing.py  # 16kHz resampling & 5s segmentation
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

## ⚙️ Installation & Environment Setup

Ensure you have Python 3.8+ installed. We highly recommend using a virtual environment (e.g., Conda). Install the required dependencies:

    pip install -r requirements.txt

*(Note: Ensure `torch`, `lightning`, `librosa`, and `torchmetrics` are properly installed and compatible with your CUDA version for GPU acceleration).*

## 📊 Dataset Preparation & Evaluation Protocol

We utilize the widely recognized public **ShipsEar** dataset.

1. Download the raw audio files from the official ShipsEar repository.
2. Place the audio files into folders `A`, `B`, `C`, `D`, and `E` corresponding to their respective vessel classes inside a root directory named `shipsEar_AUDIOS/`.
3. The preprocessing script (`ShipsEar_Data_Preprocessing.py`) performs 16 kHz resampling and non-overlapping 5-second segmentation.

**⚠️ Evaluation Protocol (Crucial for preventing data leakage):**
To ensure rigorous evaluation and prevent data leakage, train/validation/test splits **must be created at the recording level** rather than randomly splitting over segments. Splitting segments randomly across sets can lead to adjacent segments from the same continuous recording appearing in both training and testing sets, artificially inflating performance. Please ensure your `split_indices.txt` reflects a recording-level split.

## 🏃‍♂️ How to Run

To train the HTAN model from scratch, execute the main script.
We recommend using a batch size of `64` and an initial learning rate of `1e-3` for training from scratch.

    python demo_light.py \
        --model HTAN \
        --data_selection 1 \
        --train_batch_size 64 \
        --lr 1e-3 \
        --num_epochs 100 \
        --audio_feature LogMelFBank \
        --number_mels 128

### Monitoring Training
This project integrates PyTorch Lightning's TensorBoard logger. You can monitor the training progress, validation accuracy (`val_acc`), and Area Under the Precision-Recall Curve (`val_auprc`) in real-time by running:

    tensorboard --logdir=tb_logs/

## 🔬 Ablation Studies (For Paper Validation)

To validate the acoustic physical assumptions proposed in our paper, you can conduct ablation studies. We recommend implementing parameter switches in your configuration to easily toggle these features:

- **w/o Physics Prior (Pure Data-Driven Graph):** Disable the prior masking step in `HarmonicFrequencyGCN.forward()` (e.g., set a `use_prior_mask=False` flag).
- **w/o Temporal Evolution:** Replace the `BiGRU` temporal encoder with a simple temporal mean pooling operation (e.g., set a `use_temporal_encoder=False` flag).
- **w/o Multi-Scale Frontend:** Remove `branch2` (1x7) and `branch3` (7x1) in the `MultiScaleConvBlock`, reverting it to a standard local CNN.
- **w/o Context Branch:** Disable `branch4_pool` to evaluate the impact of the lightweight global SE-style context bias (e.g., set a `use_context_branch=False` flag).

## 📈 Results (Placeholder)

*(Fill this section with your final experimental results to provide a quick overview of model performance.)*

| Model Variant | Params (M) | FLOPs (G) | Accuracy (%) | Macro-F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (AST Fine-tuned) | ~86.0 | - | - | - |
| HTAN (w/o Prior Mask) | ~1.5 | - | - | - |
| HTAN (Full) | ~1.5 | - | **XX.X** | **XX.X** |

---
## 📄 Citation

*If you find this code or our physics-inspired methodology helpful in your research, please consider citing our work:*

    @article{YourName2026HTAN,
      title={Robust Underwater Acoustic Target Recognition with Physics-Inspired Harmonic-Temporal Attention Network},
      author={Your Name and Co-authors},
      journal={TBD (e.g., Ocean Engineering / IEEE JOE)},
      year={2026}
    }