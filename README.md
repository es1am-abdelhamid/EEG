# Subject-Calibrated Feature and Functional Connectivity Fusion with Hierarchical Meta-Ensemble Learning for Reduced-Channel EEG Emotion Recognition

<p align="center">
  <a href="https://github.com/es1am-abdelhamid/EEG">
    <img src="https://img.shields.io/badge/Repository-EEG%20Emotion%20Recognition-blue?logo=github" alt="GitHub Repository">
  </a>
  <a href="https://www.mathworks.com/products/matlab.html">
    <img src="https://img.shields.io/badge/MATLAB-R2021a%2B-orange?logo=mathworks" alt="MATLAB">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-GPU%20Accelerated-ee4c2c?logo=pytorch" alt="PyTorch">
  </a>
  <a href="https://colab.research.google.com/">
    <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab" alt="Google Colab">
  </a>
  <a href="https://orcid.org/0009-0000-6822-1182">
    <img src="https://img.shields.io/badge/ORCID-0009--0000--6822--1182-a6ce39?logo=orcid" alt="ORCID">
  </a>
</p>

**Official source code and reproducibility repository** for the research study:

*Subject-Calibrated Feature and Functional Connectivity Fusion with Hierarchical Meta-Ensemble Learning for Reduced-Channel EEG Emotion Recognition*

**Author**
- **Eslam A. Abd El-Razek** — First and Corresponding Author

**Affiliation:** College of Engineering, Arab Academy for Science, Technology and Maritime Transport (AASTMT), Cairo, Egypt.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Repository Structure](#repository-structure)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Multi-Domain Feature Extraction](#multi-domain-feature-extraction)
- [Statistical Channel Selection](#statistical-channel-selection)
- [Subject-Calibrated Feature Engineering](#subject-calibrated-feature-engineering)
- [Functional Connectivity and Feature Fusion](#functional-connectivity-and-feature-fusion)
- [Hierarchical Meta-Ensemble Classification](#hierarchical-meta-ensemble-classification)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Expected Outputs](#expected-outputs)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This repository provides the MATLAB and Python/PyTorch implementation of a multi-stage EEG emotion recognition framework designed to distinguish three discrete emotional states:

- **Amusement** — Emotion ID 4
- **Confusion** — Emotion ID 11
- **Fear** — Emotion ID 17

The proposed framework combines multi-domain EEG feature extraction, statistically guided channel reduction, subject-calibrated feature engineering, functional connectivity analysis, Riemannian covariance descriptors, and hierarchical ensemble learning.

Experiments use the `EEGEmotions-27` corpus, with recordings sampled at 256 Hz from 14 EEG channels. After preprocessing and class balancing, the working dataset contains **1,305 EEG segments**, with 435 segments per emotion.

The framework reduces the original 14-channel montage to six selected channels and integrates complementary single-channel and inter-channel representations into a fused feature space containing **3,570 features per segment**.

The final classification stage evaluates heterogeneous machine learning and deep learning models using five-fold stratified cross-validation.

**Repository:** [es1am-abdelhamid/EEG](https://github.com/es1am-abdelhamid/EEG)

## Key Contributions

### 1. Statistical EEG Channel Reduction

A five-metric statistical consensus ranking evaluates 2,240 features extracted from the original 14-channel EEG recordings. The resulting reduced montage contains six channels:

\[
\mathcal{G}=\{10,5,6,7,3,8\}.
\]

This corresponds to a reduction from 14 channels to 6 channels, or approximately **57.1% fewer electrodes**.

### 2. Subject-Calibrated Feature Engineering

A three-part feature representation combines spatial normalization, cross-channel differential descriptors, and participant-specific robust calibration to reduce inter-subject distributional differences.

### 3. Functional Connectivity Fusion

The framework incorporates Phase-Locking Value (PLV), Phase-Lag Index (PLI), instantaneous envelope correlation, magnitude-squared coherence, and Riemannian covariance descriptors across the 15 unique pairs formed by six channels.

### 4. Leakage-Controlled Classification

Training-dependent preprocessing, feature selection, and hyperparameter selection are designed to operate within the relevant cross-validation folds to reduce information leakage.

### 5. Hierarchical Meta-Ensemble Learning

Multiple heterogeneous classifiers are combined with a two-stage specialist routing strategy. The routing mechanism first separates Amusement from the remaining emotions and then focuses on the Confusion-versus-Fear distinction.

---

### Reported Best Standalone Models

| Classification Task | Best Reported Base Learner | Accuracy |
|---|---|---:|
| Amusement vs. Confusion | Tabular ResNet with Squeeze-and-Excitation | 83.76 ± 2.48% |
| Amusement vs. Fear | Multi-Kernel Cholesky Classifier | 85.56 ± 3.58% |
| Confusion vs. Fear | Squared-Inverse Correlation k-NN | 81.34 ± 3.91% |
| Three-class classification | Tuned XGBoost | 80.11 ± 2.73% |

The pairwise classification experiments use 870 samples per task, corresponding to two balanced classes with 435 segments each.

---

## End-to-End Pipeline

The complete workflow is divided into five main processing stages, preceded by dataset preprocessing.

| Stage | Script | Input | Output | Main Operation |
|---|---|---|---|---|
| 0 | `data_preprocessing.m` | Raw 14-channel EEG | 1,305 × 14 × 2,560 | Filtering, artifact processing, segmentation, and class balancing |
| 1–2 | `2240_FEATURE_EXTRACTION.m` | Preprocessed EEG | 1,305 × 2,240 | Multi-domain feature extraction and statistical channel selection |
| 3 | `SPATIAL_ASYMMETRY_SUBJECT_BASELINE_CALIBRATION.m` | Selected-channel features | 1,305 × 2,880 | Spatial normalization, differential features, and subject calibration |
| 4 | `CONNECTIVITY_FUSION.m` | Calibrated features and selected-channel EEG | 1,305 × 3,570 | Connectivity extraction and feature fusion |
| 5 | `HIERARCHICAL_META-ENSEMBLE_PIPELINE.py` | Fused feature matrix | OOF predictions, metrics, and plots | Feature selection, model training, and hierarchical ensemble classification |

The dimensions above describe the intended data flow. Confirm the actual array shapes and generated filenames against the scripts supplied in the repository.

---

## Repository Structure

The principal scripts are organized by processing stage.

```text
EEG/
├── README.md
├── data_preprocessing.m
├── 2240_FEATURE_EXTRACTION.m
├── SPATIAL_ASYMMETRY_SUBJECT_BASELINE_CALIBRATION.m
├── CONNECTIVITY_FUSION.m
└── HIERARCHICAL_META-ENSEMBLE_PIPELINE.py
```

Additional generated data, diagnostic plots, Excel reports, and result archives may be created when the scripts are executed.

The filenames above describe the expected main pipeline. Generated artifacts and optional supporting files are not necessarily committed to the repository.

---

## Dataset and Preprocessing

**Dataset:** `EEGEmotions-27`

**Sampling frequency:** 256 Hz

**Original montage:** 14 EEG channels

**Window duration:** 10 seconds

**Samples per window:**

\[
L=f_sT=256\times10=2,560.
\]

### Target Emotion Classes

| Emotion | Emotion ID | Number of Segments |
|---|---:|---:|
| Amusement | 4 | 435 |
| Confusion | 11 | 435 |
| Fear | 17 | 435 |
| **Total** | — | **1,305** |

### Preprocessing Operations

The preprocessing stage is intended to:

1. Load the raw multichannel EEG recordings.
2. Apply band-pass filtering over 0.5–45 Hz.
3. Perform the artifact-processing procedure implemented in the MATLAB script.
4. Segment recordings into 10-second windows containing 2,560 samples per channel.
5. retain the three target emotion categories.
6. Balance the classes to obtain 435 segments per emotion.
7. Save the processed dataset as `eeg_41711.mat`.

The exact filtering implementation, artifact-removal method, segment-overlap policy, and participant-level inclusion criteria should be checked in `data_preprocessing.m` when reproducing the study.

---

## Multi-Domain Feature Extraction

The feature extraction stage constructs a **2,240-dimensional feature matrix**, with 160 descriptors per channel across 14 channels.

\[
14\times160=2,240.
\]

The four feature groups are designed to capture complementary temporal, spectral, and time-frequency properties of EEG signals.

### 1. Time-Domain and Nonlinear Features

**40 features per channel**

The reported descriptors include:

- Amplitude and distribution statistics
- Hjorth parameters
- Zero-crossing rate
- Teager–Kaiser energy operator (TKEO)
- Permutation entropy
- Higuchi, Petrosian, and Katz fractal dimensions
- Autocorrelation features
- Subwindow-based nonstationarity indices

### 2. Frequency-Domain Features

**40 features per channel**

Welch power spectral density estimates are used to derive absolute, relative, and logarithmic band-power descriptors.

| Frequency Band | Frequency Range |
|---|---|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

Additional descriptors include cross-band power ratios, spectral entropy, spectral edge frequency (SEF95), and spectral moments.

### 3. Daubechies-4 Wavelet Features

**40 features per channel**

A five-level discrete wavelet decomposition using `db4` extracts information from approximation and detail coefficients:

\[
A_5,\ D_5,\ D_4,\ D_3,\ D_2,\ D_1.
\]

The feature set includes sub-band energies, RMS values, relative energy ratios, skewness, kurtosis, and wavelet entropy measures.

### 4. Coiflet-5 Wavelet Features

**40 features per channel**

The `coif5` decomposition provides a complementary multiresolution representation of EEG transients and signal morphology.

The extracted descriptors follow the wavelet feature framework implemented in the MATLAB script.

---

## Statistical Channel Selection

Channel selection is performed using a five-metric statistical consensus ranking.

### Statistical Criteria

| Criterion | Weight |
|---|---:|
| Direction-invariant ROC-AUC | 0.25 |
| Absolute Pearson correlation | 0.20 |
| One-way ANOVA F-statistic | 0.20 |
| Absolute Mann–Whitney Z-statistic | 0.20 |
| Kolmogorov–Smirnov statistic | 0.15 |
| **Total** | **1.00** |

The framework also reports the three-class Kruskal–Wallis H-statistic as an additional diagnostic measure.

The consensus procedure identifies the following six channels:

\[
\mathcal{G}=\{10,5,6,7,3,8\}.
\]

The reported channel labels are:

- `Channel_10`
- `Channel_05`
- `Channel_06`
- `Channel_07`
- `Channel_03`
- `Channel_08`

This reduces the montage from 14 electrodes to 6, decreasing the channel count by approximately 57.1%.

**Expected diagnostic outputs:**

- `Master_Summary_Top100_All_Tests.xlsx`
- `Master_Summary_Diagnostic_2240Features.png`

The channel ranking is statistically guided. Any feature-selection operation that uses emotion labels during predictive evaluation must be restricted to the relevant training fold to avoid information leakage.

---

## Subject-Calibrated Feature Engineering

The calibration stage is designed to reduce subject-specific baseline differences and incorporate spatial relationships between the selected channels.

The intended final representation contains **2,880 calibrated features**, organized into three groups.

### Set 1 — Within-Segment Spatial Relative Normalization

**960 features**

For each segment, the selected-channel feature values are normalized relative to the spatial distribution across the six selected channels.

The general form is:

\[
z_{c,j}=
\frac{x_{c,j}-\mu_j^{\mathrm{spatial}}}
{\sigma_j^{\mathrm{spatial}}+\epsilon},
\]

where \(c\) denotes the channel, \(j\) denotes the feature index, and \(\epsilon\) is a small numerical-stability constant.

### Set 2 — Cross-Channel Differential Asymmetry

**Reported: 960 features**

This component is intended to encode directional differences and normalized spatial contrasts between selected electrodes.

The supplied project description refers to 12 neurophysiologically meaningful electrode pairs. Because six channels produce 15 unique unordered pairs, the exact pair definition and the resulting feature count must be verified against the MATLAB implementation.

### Set 3 — Subject-Baseline Calibration (`SubjCal`)

**960 features**

The intended calibration uses robust participant-specific statistics, including median centering and interquartile-range scaling.

A general robust normalization is:

\[
z_{s,j}=
\frac{x_{s,j}-\operatorname{median}(x_{s,j})}
{\operatorname{IQR}(x_{s,j})/1.349+\epsilon},
\]

where \(s\) identifies the participant and \(j\) identifies a feature.

This transformation can reduce between-subject distributional differences when the reference observations are available under a valid evaluation protocol.

**Expected outputs:**

- `EEG_Step3_Calibrated_Data.mat`
- `Step2_Calibrated_Statistical_Evaluation.xlsx`

**Reproducibility note:** Subject calibration must use only information legitimately available at inference time. If calibration statistics are computed using held-out samples or labels unavailable during deployment, the evaluation protocol must explicitly address that dependency.

---

## Functional Connectivity and Feature Fusion

The connectivity stage extends the six-channel representation by analyzing interactions between the selected electrodes.

For six channels, the number of unique unordered channel pairs is:

\[
\binom{6}{2}=\frac{6(6-1)}{2}=15.
\]

The implementation description reports 345 connectivity descriptors per segment, followed by a corresponding 345-feature calibrated connectivity representation.

### Connectivity Feature Groups

| Feature Group | Reported Features | Description |
|---|---:|---|
| Phase-Locking Value (PLV) | 75 | Phase synchronization across channel pairs and frequency bands |
| Phase-Lag Index (PLI) | 75 | Phase-lag asymmetry intended to reduce sensitivity to zero-lag coupling |
| Envelope Correlation (`EnvCorr`) | 75 | Correlation between instantaneous amplitude envelopes |
| Magnitude-Squared Coherence (`MSCoh`) | 75 | Frequency-domain linear coupling |
| Riemannian covariance and eigen-synchrony descriptors | 45 | Covariance geometry, eigenvalue, and related summary descriptors |
| **Total raw connectivity** | **345** | — |

The four band-specific connectivity groups each contain 75 descriptors, corresponding to 15 channel pairs and five frequency bands.

### Riemannian Covariance Descriptors

The reported representation includes shrinkage-regularized covariance matrices and matrix-logarithm-based descriptors.

For a symmetric positive-definite covariance matrix \(\hat{\mathbf C}_{\mathrm{SPD}}\), a log-Euclidean representation can be expressed as:

\[
\mathbf L=\log(\hat{\mathbf C}_{\mathrm{SPD}}).
\]

The implementation description also includes upper-triangular matrix descriptors, log-eigenvalue information, spectral entropy, and a Frobenius-norm summary.

The precise reference matrix and tangent-space interpretation should follow the mathematical implementation in `CONNECTIVITY_FUSION.m`.

### Final Feature Fusion

The intended feature composition is:

| Component | Number of Features |
|---|---:|
| Subject-calibrated single-channel features | 2,880 |
| Raw functional connectivity features | 345 |
| Calibrated functional connectivity features | 345 |
| **Total fused features** | **3,570** |

Thus, the fused matrix is intended to have the form:

\[
\mathbf F_{\mathrm{fused}}
\in\mathbb R^{1305\times3570}.
\]

**Expected output:**

`EEG_Step4_Connectivity_Fused_Data.mat`

The raw and calibrated connectivity components are described as complementary representations of inter-channel relationships.

---

## Hierarchical Meta-Ensemble Classification

The final classification stage is implemented in Python using machine learning libraries, PyTorch, and GPU acceleration where supported.

The pipeline is designed around five-fold stratified cross-validation, fold-specific preprocessing, feature selection, and model evaluation.

### 1. Fold-Specific Preprocessing

The reported procedure includes:

- Training-fold z-score standardization
- Outlier clipping at ±6 standard deviations
- Hybrid relevance scoring
- Quota-balanced feature candidate selection
- Correlation-penalized greedy minimum-redundancy maximum-relevance (mRMR) selection

The candidate pool uses the following quotas:

| Feature Domain | Candidate Quota |
|---|---:|
| Single-channel features | 200 |
| Connectivity features | 80 |
| **Total** | **280** |

The reported hybrid relevance score combines ANOVA F-statistics and rank-based AUC information, with weights of 0.55 and 0.45, respectively.

The correlation-redundancy penalty is reported as:

\[
\lambda_{\mathrm{red}}=0.42.
\]

### 2. Dual-Subspace Routing

The pipeline selects different feature-subspace sizes for different model families.

| Subspace | Candidate Values | Intended Model Families |
|---|---|---|
| Compact | 20, 28, 35, 45 | Distance-based and kernel learners |
| Rich | 45, 60, 80, 100 | Tree-based and deep learners |

The optimal subspace size is selected using the reported nested inner cross-validation procedure.

### 3. Base Learners

The supplied configuration describes six heterogeneous base learners.

#### GPU XGBoost

`GPU_XGBoost_Tuned`

A regularized histogram-based gradient-boosted tree classifier.

Reported settings include:

- Maximum number of boosting rounds: 420
- Maximum tree depth: 4
- Learning rate: 0.035
- L1 regularization: 0.2
- L2 regularization: 1.5
- Validation-based early stopping

#### GPU Tabular ResNet with Squeeze-and-Excitation

`GPU_Tabular_ResNet_SE`

A deep tabular residual network incorporating:

- Squeeze-and-Excitation feature attention
- Layer normalization
- SiLU activation
- Inverted dropout with \(p=0.25\)
- AdamW optimization
- Cosine annealing
- Label smoothing with \(\varepsilon=0.04\)

#### GPU Multi-Kernel Cholesky Classifier

`GPU_MultiKernel_Cholesky`

A kernel-based classifier combining multiscale RBF and linear/quadratic cosine-kernel representations.

The reported implementation uses Nyström approximation with up to 120 landmarks and batched Cholesky factorization for the associated regularized linear systems.

#### Tuned RBF Support Vector Machine

`Tuned_RBF_SVM`

An RBF-kernel support vector machine with nested three-fold grid search and probability calibration.

#### Squared-Inverse Correlation k-NN

`SquaredInv_Corr_KNN`

A correlation-distance nearest-neighbor classifier evaluated using:

\[
K\in\{7,11,15\}.
\]

#### Bagged Random Forest and Extra Trees

`Bagged_ExtraTrees_RF`

A reported soft-voting combination of 300 Random Forest trees and 300 Extra Trees estimators.

### 4. Hierarchical Meta-Ensemble

`Hierarchical_T4_Meta_Ensemble`

The final ensemble combines predictions from heterogeneous learners and a two-stage specialist routing mechanism.

The reported three-class fusion rule is:

\[
\mathbf P_{\mathrm{final}}
=
0.55\mathbf P_{\mathrm{router}}
+
0.45\mathbf P_{\mathrm{flat}},
\]

where:

- \(\mathbf P_{\mathrm{router}}\) represents the routed specialist predictions.
- \(\mathbf P_{\mathrm{flat}}\) represents the flat multiclass ensemble predictions.

The specialist routing strategy is described as follows:

1. **Stage 1:** Distinguish Amusement from the remaining emotional states.
2. **Stage 2:** Resolve the Confusion-versus-Fear classification boundary.
3. **Final fusion:** Combine the routed and flat predictions according to the reported weighting scheme.

The exact probability construction, class ordering, and fallback behavior should be confirmed against the Python implementation.

---

## Hardware and Software Requirements

### Hardware Used in the Study

| Component | Reported Environment |
|---|---|
| Local CPU | 6th-generation Intel Core i7 H-series processor |
| Local GPU | NVIDIA Quadro M2000 with 4 GB GDDR5 memory |
| Cloud platform | Google Colaboratory Free Tier |
| Cloud GPU | NVIDIA Tesla T4 with 16 GB GDDR6 memory |

The MATLAB stages are intended to run on the local workstation, while the classification pipeline can use the Colab T4 runtime or a compatible local CUDA environment.

The actual hardware requirements depend on the selected algorithms, batch sizes, installed library versions, and runtime configuration.

### MATLAB Requirements

**MATLAB R2021a or later**

Required toolboxes, according to the supplied implementation description:

- Signal Processing Toolbox
- Wavelet Toolbox
- Statistics and Machine Learning Toolbox

### Python Requirements

**Python 3.9 or later**

The supplied Python pipeline uses the following packages:

- PyTorch
- NumPy
- SciPy
- pandas
- scikit-learn
- XGBoost
- openpyxl
- Matplotlib
- seaborn

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/es1am-abdelhamid/EEG.git
cd EEG
```

### 2. Create a Python Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install the Python Dependencies

```bash
python -m pip install --upgrade pip
pip install torch numpy scipy pandas scikit-learn xgboost openpyxl matplotlib seaborn
```

For GPU acceleration, install a PyTorch build compatible with the CUDA environment available on your machine. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Reproducing the Experiments

Run the stages in the following order.

### Stage 0 — EEG Preprocessing

Open MATLAB in the repository root and execute:

```matlab
run('data_preprocessing.m');
```

This stage prepares and balances the target emotion segments.

### Stages 1–2 — Feature Extraction and Channel Selection

```matlab
run('2240_FEATURE_EXTRACTION.m');
```

This stage extracts the 2,240-dimensional multi-domain representation and generates the statistical channel-selection outputs.

### Stage 3 — Subject Calibration

```matlab
run('SPATIAL_ASYMMETRY_SUBJECT_BASELINE_CALIBRATION.m');
```

This stage constructs the calibrated single-channel feature representation.

### Stage 4 — Connectivity and Feature Fusion

```matlab
run('CONNECTIVITY_FUSION.m');
```

This stage computes functional connectivity descriptors and constructs the fused feature matrix.

### Stage 5 — Classification and Meta-Ensemble Evaluation

Run the Python pipeline after verifying that the required fused MATLAB output is available at the path expected by the script.

```bash
python HIERARCHICAL_META-ENSEMBLE_PIPELINE.py
```

For Google Colab, upload or mount the required input data and Python script, select a compatible GPU runtime, and ensure that all paths and dependencies are configured correctly.

**Important:** The commands above assume that the scripts use the stated filenames and can locate their input/output files from the repository root. If a script uses hard-coded paths, update those paths before execution.

---

## Expected Outputs

The following artifacts are reported by the project description.

| Artifact | Description |
|---|---|
| `eeg_41711.mat` | Preprocessed and balanced EEG segments |
| `Master_Summary_Top100_All_Tests.xlsx` | Statistical feature-ranking summary |
| `Master_Summary_Diagnostic_2240Features.png` | Diagnostic visualization for feature selection |
| `EEG_Step3_Calibrated_Data.mat` | Calibrated single-channel feature matrix |
| `Step2_Calibrated_Statistical_Evaluation.xlsx` | Calibration evaluation report |
| `EEG_Step4_Connectivity_Fused_Data.mat` | Fused 3,570-dimensional feature matrix |
| `Step5_T4_GPU_3570Fused_Final_Report.xlsx` | Classification benchmark report |

The Python stage is also described as exporting out-of-fold confusion matrices, performance plots, and a ZIP archive containing experimental results.

Actual filenames and export behavior depend on the code version and output settings.

---

## Reproducibility and Evaluation Notes

For reliable scientific reproduction, the following implementation details should be documented and verified:

1. **Participant-level separation:** Determine whether cross-validation folds are split by segment or by participant. Segment-level stratification alone does not establish subject-independent generalization.
2. **Calibration availability:** Ensure participant-specific statistics are computed only from data permitted by the evaluation and intended deployment protocol.
3. **Fold-specific feature selection:** Any supervised feature ranking, mRMR selection, or hyperparameter tuning must be fitted within the appropriate training partition.
4. **Metric definitions:** Report macro-averaging conventions, class ordering, OOF probability construction, and whether ROC-AUC is calculated from pooled OOF predictions or averaged fold-wise.
5. **Feature dimensions:** Verify the reported 960-feature differential block, the 12 selected electrode pairs, and the total 2,880-feature calibration matrix against the implementation.
6. **Model count:** The described architecture contains six base learners plus a meta-ensemble. These should be reported separately when counting evaluated models.
7. **Software versions and random seeds:** Record MATLAB and Python package versions, random seeds, GPU settings, and the exact dataset version.

These checks are particularly important when interpreting performance estimates from overlapping EEG segments or subject-calibrated representations.

---

## Citation

If you use this repository or build upon the proposed methodology, please cite the associated research work.

The manuscript is described as under review. Replace the placeholder publication information below with the final bibliographic details when available.

```bibtex
@article{abdelrazek2026eeg,
  author  = {Abd El-Razek, Eslam A. and El-Badawy, Ismail M.},
  title   = {Subject-Calibrated Feature and Functional Connectivity Fusion
             with Hierarchical Meta-Ensemble Learning for Reduced-Channel
             {EEG} Emotion Recognition},
  journal = {Under Review},
  year    = {2026},
  url     = {https://github.com/es1am-abdelhamid/EEG}
}
```

## Contact

For questions about the implementation, research collaboration, or reproducibility, contact the authors.

### Eslam A. Abd El-Razek

**First and Corresponding Author**

- **Affiliation:** Mechatronics Engineering Department, College of Engineering, Arab Academy for Science, Technology and Maritime Transport (AASTMT), Cairo, Egypt
- **Email:** [eslamelshick@gmail.com](mailto:eslamelshick@gmail.com)
- **WhatsApp:** [+20 103 477 6680](https://wa.me/201034776680)
- **ORCID:** [0009-0000-6822-1182](https://orcid.org/0009-0000-6822-1182)
- **GitHub:** [es1am-abdelhamid](https://github.com/es1am-abdelhamid)

### Ismail M. El-Badawy

**Co-author**

- **Affiliation:** Electronics and Communications Engineering Department, College of Engineering, Arab Academy for Science, Technology and Maritime Transport (AASTMT), Cairo, Egypt
- **Email:** [ismailelbadawy@aast.edu](mailto:ismailelbadawy@aast.edu)

---

<p align="center">
  <i>Research code for reduced-channel EEG emotion recognition, subject-calibrated feature learning, and functional connectivity fusion.</i>
</p>
