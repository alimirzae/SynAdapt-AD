# SynAdapt-AD: Semantic & Synthetic-Regularized Few-Shot Domain Adaptation for Acoustic Anomaly Detection

Official repository for the PhD dissertation research:
**"Few-Shot Domain Adaptation for Industrial Acoustic Anomaly Detection Using Audio-Text Foundation Models"**
*Author:* Ali Mirzaei | *Supervisor:* Dr. Saeed Pashazadeh | *University:* University of Tabriz

## 📌 Overview
`SynAdapt-AD` is a novel framework designed for unsupervised acoustic anomaly detection (ASD) under severe data scarcity (K <= 10 normal target samples) and domain shift conditions (machine speed variations, background factory noise, smartphone capture).

### Key Features
- **Order-Domain Resampling:** Eliminates spectral smearing caused by drill dimmer/speed variations.
- **Dynamic Gated Timbral Fusion:** Fuses 512-D CLAP semantic embeddings with a 7-D perceptual timbral feature vector (Sharpness, Roughness, Shimmer, etc.).
- **Synthetic Boundary Regularization:** Uses `AudioLDM-2` generated anomaly samples with `Hinge Boundary Loss` to freeze decision boundaries.
- **PEFT & CoOp Adaptation:** Parameter-Efficient Fine-Tuning (d_mid=64) with Context Optimization for rapid calibration.
- **Noise & Sensor Robustness:** Volume gain sensitivity testing and CMVN preprocessing for mobile microphone capture.

## 🛠️ Installation & Setup
```bash
git clone https://github.com/alimirzae/SynAdapt-AD.git
cd SynAdapt-AD
pip install -r requirements.txt
```

## 🚀 Pipeline Workflow
1. `notebooks/01_data_prep_order_domain.ipynb` - Signal pre-processing & order tracking.
2. `notebooks/02_feature_caching_clap.ipynb` - Extracting and caching CLAP + timbral embeddings.
3. `notebooks/03_audioldm2_synthetic_gen.ipynb` - Synthesizing anomalous audio & quality filtering (FAD & TDI).
4. `notebooks/04_peft_coop_training.ipynb` - Training bottleneck adapters with Hinge Boundary Loss.
5. `notebooks/05_evaluation_robustness.ipynb` - Robustness benchmarks (SNR -6dB, Volume Gain Test).
6. `notebooks/06_paper_export_markdown.ipynb` - Exporting LaTeX and Markdown tables for papers.
