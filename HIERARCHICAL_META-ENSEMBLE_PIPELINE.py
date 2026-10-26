#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# STEP 5: T4 GPU ACCELERATED FUSED 3570-FEATURE (SINGLE-CHANNEL + CONNECTIVITY)
#         CLASSIFICATION & HIERARCHICAL META-ENSEMBLE PIPELINE
# File: step3_t4_gpu_3fold_pipeline.py (Modified for Fused 3,570 Features)
# =============================================================================

import os
import time
import tracemalloc
import zipfile
import warnings

import h5py
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =============================================================================
# 1. PURE-TENSOR BATCHED CHOLESKY SOLVER (BYPASSES CUDA 13.0 MAGMA BUG ON T4)
# =============================================================================
def batched_spd_cholesky_solve(A, B_rhs):
    """
    Solves A * W = B_rhs for Symmetric Positive Definite (SPD) matrices
    A: [B, D, D] and B_rhs: [B, D, C] using pure elementwise tensor ops.
    100% immune to `cudaErrorNoKernelImageForDevice` on Tesla T4 (sm_75) in cu130.
    """
    batch_size, D, _ = A.shape
    L = torch.zeros_like(A)

    for i in range(D):
        for j in range(i + 1):
            if j == 0:
                s = A[:, i, j]
            else:
                s = A[:, i, j] - torch.sum(L[:, i, :j] * L[:, j, :j], dim=1)
            if i == j:
                L[:, i, i] = torch.sqrt(torch.clamp(s, min=1e-7))
            else:
                L[:, i, j] = s / L[:, j, j]

    Z = torch.zeros_like(B_rhs)
    for i in range(D):
        if i == 0:
            Z[:, i, :] = B_rhs[:, i, :] / L[:, i, i].unsqueeze(1)
        else:
            sum_lz = torch.sum(L[:, i, :i].unsqueeze(2) * Z[:, :i, :], dim=1)
            Z[:, i, :] = (B_rhs[:, i, :] - sum_lz) / L[:, i, i].unsqueeze(1)

    W = torch.zeros_like(B_rhs)
    for i in range(D - 1, -1, -1):
        if i == D - 1:
            W[:, i, :] = Z[:, i, :] / L[:, i, i].unsqueeze(1)
        else:
            sum_ltw = torch.sum(L[:, i+1:, i].unsqueeze(2) * W[:, i+1:, :], dim=1)
            W[:, i, :] = (Z[:, i, :] - sum_ltw) / L[:, i, i].unsqueeze(1)

    return W


# =============================================================================
# 2. T4 GPU HARDWARE VERIFICATION
# =============================================================================
def verify_t4_gpu():
    if not torch.cuda.is_available():
        return False, torch.device('cpu'), 'CPU_Fallback (No CUDA Device)'
    try:
        dev = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        X_t = torch.ones((32, 24, 10), device=dev, dtype=torch.float32)
        eye_t = 1.0 * torch.eye(10, dtype=torch.float32).to(dev).unsqueeze(0)
        A_t = torch.bmm(X_t.transpose(1, 2), X_t) + eye_t
        B_t = torch.ones((32, 10, 3), device=dev, dtype=torch.float32)
        W_t = batched_spd_cholesky_solve(A_t, B_t)
        torch.cuda.synchronize()
        _ = float(W_t[0, 0, 0].cpu().item())
        return True, dev, f"CUDA_GPU ({gpu_name})"
    except Exception as e:
        return False, torch.device('cpu'), f"CPU_Fallback ({str(e)[:40]})"


def verify_xgb_gpu():
    if not torch.cuda.is_available():
        return False, 'CPU_Fallback'
    try:
        gpu_name = torch.cuda.get_device_name(0)
        X_d = np.random.randn(40, 8).astype(np.float32)
        y_d = np.random.randint(0, 2, size=40)
        clf = XGBClassifier(n_estimators=5, max_depth=2, tree_method='hist',
                            device='cuda', eval_metric='logloss', random_state=RANDOM_SEED)
        clf.fit(X_d, y_d)
        _ = clf.predict(X_d)
        return True, f"CUDA_GPU ({gpu_name})"
    except Exception:
        return False, 'CPU_Fallback'



FUSED_MAT_FILE = 'eeg_Fused_3570_GoldenConn_AmuConFea.mat'
OUTPUT_EXCEL   = 'Step5_T4_GPU_Fused3570_Final_Results.xlsx'
ZIP_FILENAME   = 'Step5_T4_GPU_Fused3570_Complete_Outputs.zip'

OUTER_FOLDS  = 5   # 5-Fold Stratified CV (set to 3 if you prefer 3-Fold CV)
INNER_FOLDS  = 3   # Inner CV for T4 GPU optimal-k search
K_COMPACT_CANDIDATES = [20, 28, 35, 45]   # For Metric/Kernel models (KNN, SVM, Cholesky)
K_RICH_CANDIDATES    = [45, 60, 80, 100]  # For Tree/Deep models (RF, ExtraTrees, XGB, ResNet)

PREV_BENCHMARKS = {
    'Amusement_vs_Confusion': {'S2_Acc': 74.25, 'S3_Acc': 72.18, 'S4_Acc': 74.37, 'S4_AUC': 0.8172},
    'Amusement_vs_Fear':      {'S2_Acc': 75.29, 'S3_Acc': 75.17, 'S4_Acc': 78.97, 'S4_AUC': 0.8607},
    'Confusion_vs_Fear':      {'S2_Acc': 67.24, 'S3_Acc': 68.16, 'S4_Acc': 69.20, 'S4_AUC': 0.7452},
    'Triple_All_3_Emotions':  {'S2_Acc': 60.15, 'S3_Acc': 62.07, 'S4_Acc': 63.07, 'S4_AUC': 0.8118}
}

TASKS = [
    {'name': 'Amusement_vs_Confusion', 'classes': [4, 11],     'labels': ['Amusement', 'Confusion']},
    {'name': 'Amusement_vs_Fear',      'classes': [4, 17],     'labels': ['Amusement', 'Fear']},
    {'name': 'Confusion_vs_Fear',      'classes': [11, 17],    'labels': ['Confusion', 'Fear']},
    {'name': 'Triple_All_3_Emotions',  'classes': [4, 11, 17], 'labels': ['Amusement', 'Confusion', 'Fear']}
]

MODEL_NAMES = [
    'GPU_XGBoost_Tuned',
    'GPU_Tabular_ResNet_SE',
    'GPU_MultiKernel_Cholesky',
    'Tuned_RBF_SVM',
    'SquaredInv_Corr_KNN',
    'Bagged_ExtraTrees_RF',
    'Hierarchical_T4_Meta_Ensemble'
]


# =============================================================================
# 4. EXACT 3,570-FEATURE METADATA BUILDER & HDF5 LOADER
# =============================================================================
def build_3570_feature_metadata():
    """
    Reconstructs exact names, channels, domains, and calibration types for all
    3,570 fused features (2,880 Step-1 features + 690 Step-4 Connectivity features).
    """
    golden_ch_idx = [10, 5, 6, 7, 3, 8]
    core_pairs = [(10, 5), (10, 6), (10, 7), (5, 6), (5, 7), (6, 7)]

    time_names = [
        'T01_Variance_Activity', 'T02_StdDev', 'T03_RMS', 'T04_MeanAbsValue', 'T05_Skewness',
        'T06_Kurtosis', 'T07_IQR', 'T08_MeanAbsDev', 'T09_MedianAbsDev', 'T10_PeakToPeak',
        'T11_MaxAbsAmp', 'T12_CrestFactor', 'T13_ShapeFactor', 'T14_ImpulseFactor', 'T15_ClearanceFactor',
        'T16_Hjorth_Mobility', 'T17_Hjorth_Complexity', 'T18_ZeroCrossRate', 'T19_SlopeSignChanges', 'T20_SecondDiff_ZCR',
        'T21_WaveformLength', 'T22_NormFirstDiff', 'T23_MeanAbsSecondDiff', 'T24_NormSecondDiff', 'T25_TKEO_Mean',
        'T26_TKEO_Std', 'T27_TKEO_Skewness', 'T28_TKEO_Kurtosis', 'T29_LogEnergy_Entropy', 'T30_ShannonEnergy_Entropy',
        'T31_PermEntropy_Lag1', 'T32_PermEntropy_Lag2', 'T33_Petrosian_FD', 'T34_Katz_FD', 'T35_Higuchi_FD',
        'T36_Autocorr_Lag1', 'T37_Autocorr_Lag2', 'T38_Autocorr_Lag4', 'T39_NonStat_Var_CV', 'T40_NonStat_MAV_CV'
    ]
    freq_names = [
        'F01_Delta_RelPow', 'F02_Theta_RelPow', 'F03_Alpha_RelPow', 'F04_Beta_RelPow', 'F05_Gamma_RelPow',
        'F06_Delta_AbsPow', 'F07_Theta_AbsPow', 'F08_Alpha_AbsPow', 'F09_Beta_AbsPow', 'F10_Gamma_AbsPow',
        'F11_Log_DeltaPow', 'F12_Log_ThetaPow', 'F13_Log_AlphaPow', 'F14_Log_BetaPow', 'F15_Log_GammaPow',
        'F16_Ratio_DeltaTheta', 'F17_Ratio_ThetaAlpha', 'F18_Ratio_AlphaBeta', 'F19_Ratio_BetaGamma', 'F20_Ratio_ThetaBeta',
        'F21_Spec_Centroid', 'F22_Spec_Spread', 'F23_Spec_Skewness', 'F24_Spec_Kurtosis', 'F25_Spec_Entropy',
        'F26_Spec_Flatness', 'F27_Spec_Rolloff85', 'F28_Spec_Bandwidth', 'F29_Peak_Frequency', 'F30_Peak_Power',
        'F31_Median_Frequency', 'F32_Mean_Frequency', 'F33_Spec_Slope', 'F34_Spec_Decrease', 'F35_Spec_Irregularity',
        'F36_Spec_CrestFactor', 'F37_Spec_EntropyRate', 'F38_Spec_Flux', 'F39_Spec_Moment1', 'F40_Spec_Moment2'
    ]
    wav_base = [
        'W01_Eng_A5', 'W02_Eng_D1', 'W03_Eng_D2', 'W04_Eng_D3', 'W05_Eng_D4', 'W06_Eng_D5',
        'W07_EngRatio_A5', 'W08_EngRatio_D1', 'W09_EngRatio_D2', 'W10_EngRatio_D3', 'W11_EngRatio_D4', 'W12_EngRatio_D5',
        'W13_ShanEnt_A5', 'W14_ShanEnt_D1', 'W15_ShanEnt_D2', 'W16_ShanEnt_D3', 'W17_ShanEnt_D4', 'W18_ShanEnt_D5',
        'W19_RMS_A5', 'W20_RMS_D1', 'W21_RMS_D2', 'W22_RMS_D3', 'W23_RMS_D4', 'W24_RMS_D5',
        'W25_Kurt_A5', 'W26_Kurt_D1', 'W27_Kurt_D2', 'W28_Kurt_D3', 'W29_Kurt_D4', 'W30_Kurt_D5',
        'W31_Skew_A5', 'W32_Skew_D1', 'W33_Skew_D2', 'W34_Skew_D3', 'W35_Skew_D4', 'W36_Skew_D5',
        'W37_LogEng_Entropy', 'W38_Coeff_Sparsity', 'W39_Coeff_ZeroCrossRate', 'W40_Detail_EngDist_Entropy'
    ]
    base_160 = time_names + freq_names + [f"DB4_{w}" for w in wav_base] + [f"COIF5_{w}" for w in wav_base]
    dom_160  = ['Time_Domain']*40 + ['Frequency_Domain']*40 + ['Wavelet_db4']*40 + ['Wavelet_coif5']*40

    names, chs, doms, cals = [], [], [], []
    # Part 1: 2,880 Step-1 Features
    for ch in golden_ch_idx:
        for f_i in range(160):
            names.append(f"SpatialRel_Ch{ch:02d}_{base_160[f_i]}")
            chs.append(f"Channel_{ch:02d}")
            doms.append(dom_160[f_i])
            cals.append("1_Spatial_Relative")
    for cA, cB in core_pairs:
        for f_i in range(160):
            names.append(f"Asym_Ch{cA:02d}_vs_Ch{cB:02d}_{base_160[f_i]}")
            chs.append(f"Channel_{cA:02d}")
            doms.append(dom_160[f_i])
            cals.append("2_CrossChannel_Asymmetry")
    for ch in golden_ch_idx:
        for f_i in range(160):
            names.append(f"SubjCal_Ch{ch:02d}_{base_160[f_i]}")
            chs.append(f"Channel_{ch:02d}")
            doms.append(dom_160[f_i])
            cals.append("3_Subject_Calibrated")

    # Part 2: 345 Raw Connectivity + 345 Subject-Calibrated Connectivity = 690 Features
    pair_names_15 = []
    pair_anchor_ch = []
    for i in range(6):
        for j in range(i + 1, 6):
            pair_names_15.append(f"Ch{golden_ch_idx[i]:02d}_Ch{golden_ch_idx[j]:02d}")
            pair_anchor_ch.append(f"Channel_{golden_ch_idx[i]:02d}")

    band_tags = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    conn_types = ['PLV', 'PLI', 'EnvCorr', 'Coherence']
    conn_base_names, conn_base_dom, conn_base_ch = [], [], []

    for c_type in conn_types:
        for b_str in band_tags:
            for p_i in range(15):
                conn_base_names.append(f"Conn_{c_type}_{b_str}_{pair_names_15[p_i]}")
                conn_base_dom.append('Connectivity_Synchrony')
                conn_base_ch.append(pair_anchor_ch[p_i])

    for p_i in range(15):
        conn_base_names.append(f"Conn_BroadCorr_{pair_names_15[p_i]}")
        conn_base_dom.append('Connectivity_Riemannian')
        conn_base_ch.append(pair_anchor_ch[p_i])

    for c_col in range(6):
        for r_row in range(c_col + 1):
            chA = golden_ch_idx[r_row]
            chB = golden_ch_idx[c_col]
            conn_base_names.append(f"Conn_RiemannLogCov_Ch{chA:02d}_Ch{chB:02d}")
            conn_base_dom.append('Connectivity_Riemannian')
            conn_base_ch.append(f"Channel_{chA:02d}")

    for e_i in range(1, 7):
        conn_base_names.append(f"Conn_CovLogEigVal_{e_i:02d}")
        conn_base_dom.append('Connectivity_Riemannian')
        conn_base_ch.append('Channel_10')

    for extra_n in ['Conn_Global_EigEntropy', 'Conn_Global_CondRatio', 'Conn_Global_FrobeniusNorm']:
        conn_base_names.append(extra_n)
        conn_base_dom.append('Connectivity_Riemannian')
        conn_base_ch.append('Channel_10')

    for idx_c in range(345):
        names.append(f"RawConn_{conn_base_names[idx_c]}")
        chs.append(conn_base_ch[idx_c])
        doms.append(conn_base_dom[idx_c])
        cals.append("4_Connectivity_Raw")

    for idx_c in range(345):
        names.append(f"SubjConn_{conn_base_names[idx_c]}")
        chs.append(conn_base_ch[idx_c])
        doms.append(conn_base_dom[idx_c])
        cals.append("5_Connectivity_SubjCal")

    return np.array(names), np.array(chs), np.array(doms), np.array(cals)


def load_fused_3570_dataset():
    """Loads `eeg_Fused_3570_GoldenConn_AmuConFea.mat` safely via HDF5 / scipy."""
    if not os.path.exists(FUSED_MAT_FILE):
        raise FileNotFoundError(
            f"File '{FUSED_MAT_FILE}' not found! Please upload it to your Google Colab session."
        )
    print(f"Loading fused dataset '{FUSED_MAT_FILE}'...")
    try:
        with h5py.File(FUSED_MAT_FILE, 'r') as f:
            X_mat = np.array(f['Features_Fused_3570'], dtype=np.float64)
            y_emo = np.array(f['EmotionID_3']).squeeze().astype(int)
            pids  = np.array(f['ParticipantID_3']).squeeze().astype(int)
    except Exception:
        mat = sio.loadmat(FUSED_MAT_FILE)
        X_mat = np.array(mat['Features_Fused_3570'], dtype=np.float64)
        y_emo = np.array(mat['EmotionID_3']).squeeze().astype(int)
        pids  = np.array(mat['ParticipantID_3']).squeeze().astype(int)

    if X_mat.shape[0] != len(y_emo) and X_mat.shape[1] == len(y_emo):
        X_mat = X_mat.T
    X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
    names, chs, doms, cals = build_3570_feature_metadata()
    print(f"Loaded successfully! Shape: {X_mat.shape} (Segments={len(y_emo)}, Features={X_mat.shape[1]})")
    return X_mat, y_emo, pids, names, chs, doms, cals


# =============================================================================
# 5. GPU-ACCELERATED FUSED QUOTA mRMR RANKING (SINGLE-CHANNEL + CONNECTIVITY)
# =============================================================================
def rank_fused_mrmr_binary(X_tr, y_tr, dev, max_k=100):
    """
    Guarantees both Single-Channel (0..2879) and Connectivity (2880..3569)
    candidates enter the GPU correlation-penalized mRMR pool inside each fold.
    """
    n_samples, n_feats = X_tr.shape
    c1 = X_tr[y_tr == 0]
    c2 = X_tr[y_tr == 1]
    n1, n2 = len(c1), len(c2)

    mu1 = np.mean(c1, axis=0)
    mu2 = np.mean(c2, axis=0)
    mu_all = np.mean(X_tr, axis=0)
    ss_b = n1 * ((mu1 - mu_all) ** 2) + n2 * ((mu2 - mu_all) ** 2)
    ss_w = np.sum((c1 - mu1) ** 2, axis=0) + np.sum((c2 - mu2) ** 2, axis=0) + 1e-12
    f_scores = ss_b / (ss_w / (n1 + n2 - 2))
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    ranks = np.argsort(np.argsort(X_tr, axis=0), axis=0).astype(np.float64)
    r_sum = np.sum(ranks[y_tr == 1], axis=0)
    auc_v = (r_sum - (n2 * (n2 - 1)) / 2.0) / (n1 * n2)
    auc_rel = np.maximum(auc_v, 1.0 - auc_v)

    f_norm   = (f_scores - np.min(f_scores)) / (np.ptp(f_scores) + 1e-12)
    auc_norm = (auc_rel - np.min(auc_rel)) / (np.ptp(auc_rel) + 1e-12)
    rel_all  = 0.55 * f_norm + 0.45 * auc_norm

    top_single = np.argsort(rel_all[:2880])[::-1][:200]
    top_conn   = 2880 + np.argsort(rel_all[2880:])[::-1][:80]
    cand_pool  = np.concatenate([top_single, top_conn])
    rel_cand   = rel_all[cand_pool]
    rel_cand   = (rel_cand - np.min(rel_cand)) / (np.ptp(rel_cand) + 1e-12)

    X_cand_t = torch.from_numpy(X_tr[:, cand_pool].astype(np.float32)).to(dev)
    X_cand_c = X_cand_t - torch.mean(X_cand_t, dim=0, keepdim=True)
    X_cand_n = X_cand_c / (torch.sqrt(torch.sum(X_cand_c ** 2, dim=0, keepdim=True)) + 1e-8)
    corr_mat = torch.abs(torch.matmul(X_cand_n.T, X_cand_n)).cpu().numpy()
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    start_pos = int(np.argmax(rel_cand))
    selected_local = [start_pos]
    remaining = [i for i in range(len(cand_pool)) if i != start_pos]

    while len(selected_local) < min(max_k, len(cand_pool)) and remaining:
        red_vec = np.mean(corr_mat[np.ix_(remaining, selected_local)], axis=1)
        scores  = rel_cand[remaining] - 0.42 * red_vec
        best_pos = int(np.argmax(scores))
        selected_local.append(remaining[best_pos])
        remaining.pop(best_pos)

    return cand_pool[np.array(selected_local, dtype=int)]


def rank_fused_mrmr_triple(X_tr, y_tr, dev, max_k=100):
    """Interleaves Fused mRMR rankings across all 3 emotion pairs for 3-class balance."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_rankings = []
    for c1, c2 in pairs:
        m = (y_tr == c1) | (y_tr == c2)
        y_bin = (y_tr[m] == c2).astype(int)
        pair_rankings.append(rank_fused_mrmr_binary(X_tr[m], y_bin, dev, max_k=max_k))

    interleaved = []
    for step in range(max_k):
        for p_i in range(3):
            if step < len(pair_rankings[p_i]):
                f_id = int(pair_rankings[p_i][step])
                if f_id not in interleaved:
                    interleaved.append(f_id)
                if len(interleaved) >= max_k:
                    return np.array(interleaved, dtype=int)
    return np.array(interleaved, dtype=int)


# =============================================================================
# 6. MODEL ARCHITECTURES (SQUARED-INVERSE CORR KNN, GPU CHOLESKY, SE-RESNET, XGB)
# =============================================================================
def predict_squaredinv_corr_knn(X_tr, y_tr, X_te, num_cls):
    """
    Exact Python/NumPy implementation of MATLAB's winning `Squared-Inverse Correlation KNN`
    blending k_neighbors in [7, 11, 15] with w = 1 / (d_corr^2 + eps).
    """
    X_tr_c = X_tr - np.mean(X_tr, axis=1, keepdims=True)
    X_te_c = X_te - np.mean(X_te, axis=1, keepdims=True)
    X_tr_u = X_tr_c / (np.linalg.norm(X_tr_c, axis=1, keepdims=True) + 1e-12)
    X_te_u = X_te_c / (np.linalg.norm(X_te_c, axis=1, keepdims=True) + 1e-12)

    d_corr = np.clip(1.0 - np.dot(X_te_u, X_tr_u.T), 1e-6, 2.0)
    n_te = len(X_te)
    probs_accum = np.zeros((n_te, num_cls), dtype=np.float64)

    for k_nb in [7, 11, 15]:
        idx_k = np.argpartition(d_corr, kth=min(k_nb, d_corr.shape[1]-1), axis=1)[:, :k_nb]
        row_idx = np.arange(n_te)[:, None]
        d_k = d_corr[row_idx, idx_k]
        y_k = y_tr[idx_k]

        w_k = 1.0 / ((d_k ** 2) + 1e-8)
        probs_k = np.zeros((n_te, num_cls), dtype=np.float64)
        for c in range(num_cls):
            probs_k[:, c] = np.sum(w_k * (y_k == c), axis=1)
        probs_k /= (np.sum(probs_k, axis=1, keepdims=True) + 1e-12)
        probs_accum += probs_k

    probs = probs_accum / 3.0
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_gpu_multikernel_cholesky(X_tr, y_tr, X_te, num_cls, dev):
    """T4 GPU Multi-Scale RBF + Cosine + Quadratic Kernel solved via Cholesky."""
    n_tr, d = X_tr.shape
    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(dev)
    X_te_t = torch.from_numpy(X_te.astype(np.float32)).to(dev)

    tr_sq = torch.sum(X_tr_t ** 2, dim=1, keepdim=True)
    te_sq = torch.sum(X_te_t ** 2, dim=1, keepdim=True)
    dist_tr = torch.clamp(tr_sq + tr_sq.T - 2.0 * torch.matmul(X_tr_t, X_tr_t.T), min=0.0)
    dist_te = torch.clamp(te_sq + tr_sq.T - 2.0 * torch.matmul(X_te_t, X_tr_t.T), min=0.0)

    X_tr_u = X_tr_t / (torch.sqrt(tr_sq) + 1e-8)
    X_te_u = X_te_t / (torch.sqrt(te_sq) + 1e-8)
    K_cos_tr = torch.matmul(X_tr_u, X_tr_u.T)
    K_cos_te = torch.matmul(X_te_u, X_tr_u.T)

    g1, g2, g3 = 0.45 / d, 1.0 / d, 2.0 / d
    K_tr = (torch.exp(-g1 * dist_tr) + torch.exp(-g2 * dist_tr) + torch.exp(-g3 * dist_tr)) / 3.0 \
           + 0.45 * K_cos_tr + 0.20 * (K_cos_tr ** 2)
    K_te = (torch.exp(-g1 * dist_te) + torch.exp(-g2 * dist_te) + torch.exp(-g3 * dist_te)) / 3.0 \
           + 0.45 * K_cos_te + 0.20 * (K_cos_te ** 2)

    M = min(120, n_tr)
    idx_sub = np.linspace(0, n_tr - 1, M, dtype=int)
    Phi_tr = K_tr[:, idx_sub].unsqueeze(0)
    Phi_te = K_te[:, idx_sub].unsqueeze(0)

    Y_oh = np.zeros((n_tr, num_cls), dtype=np.float32)
    for c in range(num_cls):
        Y_oh[y_tr == c, c] = 1.0
    Y_tr_t = torch.from_numpy(Y_oh).to(dev).unsqueeze(0)

    reg_eye = torch.from_numpy(0.60 * np.eye(M, dtype=np.float32)).to(dev).unsqueeze(0)
    A_mat = torch.bmm(Phi_tr.transpose(1, 2), Phi_tr) + reg_eye
    B_rhs = torch.bmm(Phi_tr.transpose(1, 2), Y_tr_t)

    W_sol = batched_spd_cholesky_solve(A_mat, B_rhs)
    logits = torch.bmm(Phi_te, W_sol).squeeze(0).cpu().numpy()

    logits_s = (logits - np.max(logits, axis=1, keepdims=True)) * 3.5
    exp_l = np.exp(logits_s)
    probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-12)
    preds = np.argmax(probs, axis=1)
    return preds, probs


class SafeDropout(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask_np = (np.random.rand(*x.shape) < keep).astype(np.float32) / keep
        return x * torch.from_numpy(mask_np).to(device=x.device, dtype=x.dtype)


class TabularSEResNetGPU(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.25):
        super().__init__()
        mid_se = max(8, in_features // 3)
        self.feat_attn = nn.Sequential(
            nn.Linear(in_features, mid_se),
            nn.SiLU(),
            nn.Linear(mid_se, in_features),
            nn.Sigmoid()
        )
        self.stem = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            SafeDropout(dropout)
        )
        self.res1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            SafeDropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        self.act = nn.SiLU()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            SafeDropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        w = self.feat_attn(x)
        h = self.stem(x * (0.5 + w))
        h = self.act(h + self.res1(h))
        return self.head(h)


def predict_gpu_tabular_resnet(X_tr, y_tr, X_te, num_cls, dev, max_epochs=70, patience=14):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    sub_tr_i, val_i = next(sss.split(X_tr, y_tr))

    model = TabularSEResNetGPU(X_tr.shape[1], num_cls, dropout=0.25).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    X_sub = X_tr[sub_tr_i].astype(np.float32)
    y_sub = y_tr[sub_tr_i].astype(np.int64)
    X_val_t = torch.from_numpy(X_tr[val_i].astype(np.float32)).to(dev)
    y_val_t = torch.from_numpy(y_tr[val_i].astype(np.int64)).to(dev)

    best_loss = float('inf')
    best_state = None
    wait = 0
    n_sub = len(y_sub)

    for _ in range(max_epochs):
        model.train()
        perm = np.random.permutation(n_sub)
        for start_b in range(0, n_sub, 32):
            idx_b = perm[start_b:start_b + 32]
            bx = torch.from_numpy(X_sub[idx_b]).to(dev)
            by = torch.from_numpy(y_sub[idx_b]).to(dev)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te.astype(np.float32)).to(dev))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
    return preds, probs


def predict_gpu_xgboost(X_tr, y_tr, X_te, num_cls, xgb_gpu_ok):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    sub_tr_i, val_i = next(sss.split(X_tr, y_tr))

    clf = XGBClassifier(
        n_estimators=420,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=1.5,
        tree_method='hist',
        device='cuda' if xgb_gpu_ok else 'cpu',
        early_stopping_rounds=25,
        eval_metric='mlogloss' if num_cls > 2 else 'logloss',
        random_state=RANDOM_SEED
    )
    clf.fit(X_tr[sub_tr_i], y_tr[sub_tr_i], eval_set=[(X_tr[val_i], y_tr[val_i])], verbose=False)
    probs = clf.predict_proba(X_te)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_tuned_rbf_svm(X_tr, y_tr, X_te, num_cls):
    best_c, best_g, best_sc = 2.2, 1.0 / X_tr.shape[1], -1.0
    skf_in = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    d = X_tr.shape[1]
    for c_val in [1.2, 2.2, 4.0]:
        for g_val in [0.6 / d, 1.0 / d, 1.6 / d]:
            accs = []
            for itr, ival in skf_in.split(X_tr, y_tr):
                m = SVC(C=c_val, kernel='rbf', gamma=g_val, random_state=RANDOM_SEED)
                m.fit(X_tr[itr], y_tr[itr])
                accs.append(accuracy_score(y_tr[ival], m.predict(X_tr[ival])))
            if np.mean(accs) > best_sc:
                best_sc = np.mean(accs)
                best_c, best_g = c_val, g_val

    svm = SVC(C=best_c, kernel='rbf', gamma=best_g, probability=True, random_state=RANDOM_SEED)
    svm.fit(X_tr, y_tr)
    probs = svm.predict_proba(X_te)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_bagged_extratrees_rf(X_tr, y_tr, X_te, num_cls):
    et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                              random_state=RANDOM_SEED, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=300, max_features=0.35, min_samples_leaf=2,
                                random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    probs = 0.50 * rf.predict_proba(X_te) + 0.50 * et.predict_proba(X_te)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_specialist_3class_router_fused(X_tr_full, y_tr, X_te_full, dev):
    """
    Decoupled Pairwise Specialist Hierarchy on Fused 3570 Features:
      Stage 1: Amusement (0) vs Rest (1, 2) using Fused mRMR (compact k=28 & rich k=55)
      Stage 2: Confusion (1) vs Fear (2) using dedicated Fused mRMR (PLV/PLI/EnvCorr + EntropyRate)
    """
    y_s1 = (y_tr != 0).astype(int)
    cols_s1 = rank_fused_mrmr_binary(X_tr_full, y_s1, dev, max_k=55)
    _, p_s1_knn = predict_squaredinv_corr_knn(X_tr_full[:, cols_s1[:28]], y_s1, X_te_full[:, cols_s1[:28]], 2)
    _, p_s1_ker = predict_gpu_multikernel_cholesky(X_tr_full[:, cols_s1[:28]], y_s1, X_te_full[:, cols_s1[:28]], 2, dev)
    _, p_s1_rf  = predict_bagged_extratrees_rf(X_tr_full[:, cols_s1], y_s1, X_te_full[:, cols_s1], 2)
    p_s1 = 0.42 * p_s1_knn + 0.30 * p_s1_ker + 0.28 * p_s1_rf

    m_12 = (y_tr == 1) | (y_tr == 2)
    X_tr_12 = X_tr_full[m_12]
    y_tr_12 = (y_tr[m_12] == 2).astype(int)
    cols_s2 = rank_fused_mrmr_binary(X_tr_12, y_tr_12, dev, max_k=50)
    _, p_s2_svm = predict_tuned_rbf_svm(X_tr_12[:, cols_s2[:28]], y_tr_12, X_te_full[:, cols_s2[:28]], 2)
    _, p_s2_ker = predict_gpu_multikernel_cholesky(X_tr_12[:, cols_s2[:28]], y_tr_12, X_te_full[:, cols_s2[:28]], 2, dev)
    _, p_s2_knn = predict_squaredinv_corr_knn(X_tr_12[:, cols_s2[:28]], y_tr_12, X_te_full[:, cols_s2[:28]], 2)
    _, p_s2_rf  = predict_bagged_extratrees_rf(X_tr_12[:, cols_s2], y_tr_12, X_te_full[:, cols_s2], 2)
    p_s2 = 0.35 * p_s2_svm + 0.25 * p_s2_ker + 0.20 * p_s2_knn + 0.20 * p_s2_rf

    probs_3cls = np.column_stack([
        p_s1[:, 0],
        p_s1[:, 1] * p_s2[:, 0],
        p_s1[:, 1] * p_s2[:, 1]
    ])
    probs_3cls /= (np.sum(probs_3cls, axis=1, keepdims=True) + 1e-12)
    return probs_3cls


# =============================================================================
# 7. IN-FOLD DUAL OPTIMAL-k TUNING ON T4 GPU (k_compact & k_rich)
# =============================================================================
def tune_dual_optimal_k(X_tr_s, y_tr, ranked_cols, num_cls, dev):
    skf_in = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_k_comp, best_sc_comp = 28, -1.0
    best_k_rich, best_sc_rich = 60, -1.0
    k_records = {}

    all_k_cands = sorted(list(set(K_COMPACT_CANDIDATES + K_RICH_CANDIDATES)))
    for k in all_k_cands:
        X_k = X_tr_s[:, ranked_cols[:k]]
        acc_l, f1_l, auc_l = [], [], []
        for itr, ival in skf_in.split(X_k, y_tr):
            _, p_ker = predict_gpu_multikernel_cholesky(X_k[itr], y_tr[itr], X_k[ival], num_cls, dev)
            _, p_knn = predict_squaredinv_corr_knn(X_k[itr], y_tr[itr], X_k[ival], num_cls)
            p_b = 0.50 * p_ker + 0.50 * p_knn
            pv = np.argmax(p_b, axis=1)
            acc_l.append(accuracy_score(y_tr[ival], pv))
            f1_l.append(f1_score(y_tr[ival], pv, average='macro', zero_division=0))
            try:
                if num_cls == 2:
                    auc_l.append(roc_auc_score(y_tr[ival], p_b[:, 1]))
                else:
                    auc_l.append(roc_auc_score(y_tr[ival], p_b, multi_class='ovr', average='macro'))
            except Exception:
                auc_l.append(0.5)

        sc = 0.60 * float(np.mean(f1_l)) + 0.40 * float(np.mean(auc_l))
        k_records[k] = {'acc': float(np.mean(acc_l)), 'f1': float(np.mean(f1_l)), 'auc': float(np.mean(auc_l)), 'sc': sc}
        if k in K_COMPACT_CANDIDATES and sc > best_sc_comp:
            best_sc_comp = sc
            best_k_comp = k
        if k in K_RICH_CANDIDATES and sc > best_sc_rich:
            best_sc_rich = sc
            best_k_rich = k

    return best_k_comp, best_k_rich, k_records


# =============================================================================
# 8. VECTOR PDF CONFUSION MATRIX GENERATOR
# =============================================================================
def save_clean_confusion_matrix_pdf(y_true, y_pred, class_labels, pdf_filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0

    annot_matrix = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

    fig_size = (4.0, 3.4) if len(class_labels) == 2 else (4.8, 4.0)
    plt.figure(figsize=fig_size)

    ax = sns.heatmap(
        cm_perc, annot=annot_matrix, fmt='', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        cbar_kws={'label': 'Classification Percentage (%)'},
        annot_kws={'size': 8, 'weight': 'normal'}
    )
    plt.xlabel('Predicted Emotion Class', fontsize=8, fontweight='normal')
    plt.ylabel('True Emotion Class', fontsize=8, fontweight='normal')
    plt.xticks(fontsize=8, fontweight='normal')
    plt.yticks(fontsize=8, fontweight='normal', rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Classification Percentage (%)', fontsize=8, fontweight='normal')

    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()


# =============================================================================
# 9. MAIN EXECUTION LOOP
# =============================================================================
def main():
    pipeline_t0 = time.time()
    tracemalloc.start()

    pytorch_gpu_ok, dev, pytorch_dev_str = verify_t4_gpu()
    xgb_gpu_ok, xgb_dev_str = verify_xgb_gpu()

    print(f"PyTorch T4 GPU Engine : {pytorch_dev_str}")
    print(f"XGBoost T4 GPU Engine : {xgb_dev_str}")

    X_All, Y_All, PID_All, Feat_Names, Ch_List, Dom_List, Cal_List = load_fused_3570_dataset()

    exec_summary_rows  = []
    all_models_rows    = []
    per_fold_rows      = []
    k_search_rows      = []
    selected_feat_rows = []
    cost_rows          = []
    generated_files    = []

    best_oof_preds    = {}
    best_oof_probs    = {}
    best_y_true       = {}
    best_model_winner = {}

    all_k_cands = sorted(list(set(K_COMPACT_CANDIDATES + K_RICH_CANDIDATES)))

    for task in TASKS:
        t_name   = task['name']
        cls_ids  = task['classes']
        c_labels = task['labels']
        num_cls  = len(cls_ids)

        print(f"\n======================================================================")
        print(f"Task: {t_name} ({num_cls} Classes) | {OUTER_FOLDS}-Fold Stratified CV on {pytorch_dev_str}")
        print(f"======================================================================")

        mask_t = np.isin(Y_All, cls_ids)
        X_task = X_All[mask_t]
        y_raw  = Y_All[mask_t]

        y_task = np.zeros_like(y_raw)
        for c_i, c_id in enumerate(cls_ids):
            y_task[y_raw == c_id] = c_i
        n_samples = len(y_task)

        skf_outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        oof_preds   = {m: np.zeros(n_samples, dtype=int) for m in MODEL_NAMES}
        oof_probs   = {m: np.zeros((n_samples, num_cls), dtype=np.float64) for m in MODEL_NAMES}
        fold_accs   = {m: [] for m in MODEL_NAMES}
        fold_f1s    = {m: [] for m in MODEL_NAMES}
        fold_aucs   = {m: [] for m in MODEL_NAMES}
        model_times = {m: 0.0 for m in MODEL_NAMES}

        feat_counter = np.zeros(X_task.shape[1], dtype=int)
        chosen_k_comp_list, chosen_k_rich_list = [], []
        k_accum_acc = {k: [] for k in all_k_cands}
        k_accum_f1  = {k: [] for k in all_k_cands}
        k_accum_auc = {k: [] for k in all_k_cands}

        if pytorch_gpu_ok:
            torch.cuda.reset_peak_memory_stats()

        for fold_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(X_task, y_task), start=1):
            X_tr_raw, y_tr = X_task[tr_idx], y_task[tr_idx]
            X_te_raw, y_te = X_task[te_idx], y_task[te_idx]

            # 1) Strict In-Fold Z-Score Standardization (fit ONLY on training fold)
            std_scaler = StandardScaler()
            X_tr_s = np.clip(std_scaler.fit_transform(X_tr_raw), -6.0, 6.0)
            X_te_s = np.clip(std_scaler.transform(X_te_raw), -6.0, 6.0)

            # 2) Strict In-Fold Fused Quota mRMR Ranking (Single-Channel + Connectivity)
            if num_cls == 2:
                ranked_cols = rank_fused_mrmr_binary(X_tr_s, y_tr, dev, max_k=100)
            else:
                ranked_cols = rank_fused_mrmr_triple(X_tr_s, y_tr, dev, max_k=100)

            # 3) Dual Optimal-k Selection (k_compact for KNN/SVM/Kernel, k_rich for RF/XGB/ResNet)
            k_comp, k_rich, k_hist = tune_dual_optimal_k(X_tr_s, y_tr, ranked_cols, num_cls, dev)
            chosen_k_comp_list.append(k_comp)
            chosen_k_rich_list.append(k_rich)
            for k_v in all_k_cands:
                k_accum_acc[k_v].append(k_hist[k_v]['acc'] * 100.0)
                k_accum_f1[k_v].append(k_hist[k_v]['f1'] * 100.0)
                k_accum_auc[k_v].append(k_hist[k_v]['auc'])

            cols_comp = ranked_cols[:k_comp]
            cols_rich = ranked_cols[:k_rich]
            feat_counter[cols_rich] += 1

            X_tr_c, X_te_c = X_tr_s[:, cols_comp], X_te_s[:, cols_comp]
            X_tr_r, X_te_r = X_tr_s[:, cols_rich], X_te_s[:, cols_rich]

            # 4) Train & Predict with all 7 Models
            t0 = time.time()
            _, pr_m1 = predict_gpu_xgboost(X_tr_r, y_tr, X_te_r, num_cls, xgb_gpu_ok)
            model_times['GPU_XGBoost_Tuned'] += time.time() - t0

            t0 = time.time()
            _, pr_m2 = predict_gpu_tabular_resnet(X_tr_r, y_tr, X_te_r, num_cls, dev)
            model_times['GPU_Tabular_ResNet_SE'] += time.time() - t0

            t0 = time.time()
            _, pr_m3 = predict_gpu_multikernel_cholesky(X_tr_c, y_tr, X_te_c, num_cls, dev)
            model_times['GPU_MultiKernel_Cholesky'] += time.time() - t0

            t0 = time.time()
            _, pr_m4 = predict_tuned_rbf_svm(X_tr_c, y_tr, X_te_c, num_cls)
            model_times['Tuned_RBF_SVM'] += time.time() - t0

            t0 = time.time()
            _, pr_m5 = predict_squaredinv_corr_knn(X_tr_c, y_tr, X_te_c, num_cls)
            model_times['SquaredInv_Corr_KNN'] += time.time() - t0

            t0 = time.time()
            _, pr_m6 = predict_bagged_extratrees_rf(X_tr_r, y_tr, X_te_r, num_cls)
            model_times['Bagged_ExtraTrees_RF'] += time.time() - t0

            t0 = time.time()
            if num_cls == 2:
                pr_m7 = (0.30 * pr_m5 + 0.22 * pr_m6 + 0.20 * pr_m4 +
                         0.16 * pr_m3 + 0.12 * pr_m1)
            else:
                pr_router = predict_specialist_3class_router_fused(X_tr_s, y_tr, X_te_s, dev)
                pr_flat   = 0.34 * pr_m5 + 0.24 * pr_m4 + 0.22 * pr_m6 + 0.20 * pr_m3
                pr_m7     = 0.55 * pr_router + 0.45 * pr_flat
            model_times['Hierarchical_T4_Meta_Ensemble'] += time.time() - t0

            prob_dict = {
                'GPU_XGBoost_Tuned': pr_m1,
                'GPU_Tabular_ResNet_SE': pr_m2,
                'GPU_MultiKernel_Cholesky': pr_m3,
                'Tuned_RBF_SVM': pr_m4,
                'SquaredInv_Corr_KNN': pr_m5,
                'Bagged_ExtraTrees_RF': pr_m6,
                'Hierarchical_T4_Meta_Ensemble': pr_m7
            }

            for m_name, pr_m in prob_dict.items():
                preds_m = np.argmax(pr_m, axis=1)
                oof_preds[m_name][te_idx] = preds_m
                oof_probs[m_name][te_idx] = pr_m

                f_acc = accuracy_score(y_te, preds_m) * 100.0
                f_f1  = f1_score(y_te, preds_m, average='macro', zero_division=0) * 100.0
                try:
                    if num_cls == 2:
                        f_auc = roc_auc_score(y_te, pr_m[:, 1])
                    else:
                        f_auc = roc_auc_score(y_te, pr_m, multi_class='ovr', average='macro')
                except Exception:
                    f_auc = 0.5

                fold_accs[m_name].append(f_acc)
                fold_f1s[m_name].append(f_f1)
                fold_aucs[m_name].append(f_auc)

                per_fold_rows.append({
                    'Task_Scenario': t_name,
                    'Classifier_Name': m_name,
                    'Fold': f'Fold_{fold_idx}',
                    'k_Compact': k_comp,
                    'k_Rich': k_rich,
                    'Fold_Accuracy_Pct': round(f_acc, 2),
                    'Fold_MacroF1_Pct': round(f_f1, 2),
                    'Fold_ROC_AUC': round(f_auc, 4)
                })

            print(f"  [Fold {fold_idx}/{OUTER_FOLDS}] k=({k_comp},{k_rich}) | "
                  f"Meta-Ensemble Acc: {fold_accs['Hierarchical_T4_Meta_Ensemble'][-1]:.2f}% | "
                  f"KNN Acc: {fold_accs['SquaredInv_Corr_KNN'][-1]:.2f}%")

        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if pytorch_gpu_ok else 0.0
        _, peak_ram_b = tracemalloc.get_traced_memory()
        peak_ram_mb = peak_ram_b / (1024 ** 2)

        for k_v in all_k_cands:
            k_search_rows.append({
                'Task_Scenario': t_name,
                'Candidate_k_Features': k_v,
                'Inner_CV_Acc_Mean_Pct': round(float(np.mean(k_accum_acc[k_v])), 2),
                'Inner_CV_MacroF1_Mean_Pct': round(float(np.mean(k_accum_f1[k_v])), 2),
                'Inner_CV_AUC_Mean': round(float(np.mean(k_accum_auc[k_v])), 4)
            })

        top_f_idx = np.argsort(feat_counter)[::-1][:20]
        for r_i, f_id in enumerate(top_f_idx, start=1):
            if feat_counter[f_id] > 0:
                selected_feat_rows.append({
                    'Task_Scenario': t_name,
                    'Stability_Rank': r_i,
                    'Full_Feature_Name': Feat_Names[f_id],
                    'Channel': Ch_List[f_id],
                    'Domain': Dom_List[f_id],
                    'Calibration_Type': Cal_List[f_id],
                    'Folds_Selected': int(feat_counter[f_id]),
                    'Stability_Ratio': f"{int(feat_counter[f_id])} / {OUTER_FOLDS} Folds"
                })

        win_model = None
        win_score = -1.0

        for m_name in MODEL_NAMES:
            m_acc, s_acc = float(np.mean(fold_accs[m_name])), float(np.std(fold_accs[m_name]))
            m_f1,  s_f1  = float(np.mean(fold_f1s[m_name])),  float(np.std(fold_f1s[m_name]))
            m_auc        = float(np.mean(fold_aucs[m_name]))

            oof_p  = oof_preds[m_name]
            oof_pr = oof_probs[m_name]
            m_rec  = recall_score(y_task, oof_p, average='macro', zero_division=0) * 100.0
            m_prec = precision_score(y_task, oof_p, average='macro', zero_division=0) * 100.0
            try:
                if num_cls == 2:
                    g_auc = roc_auc_score(y_task, oof_pr[:, 1])
                else:
                    g_auc = roc_auc_score(y_task, oof_pr, multi_class='ovr', average='macro')
            except Exception:
                g_auc = m_auc

            dev_used = xgb_dev_str if 'XGB' in m_name else (pytorch_dev_str if 'GPU' in m_name else 'CPU_Multicore_+_T4_Assist')

            all_models_rows.append({
                'Task_Scenario': t_name,
                'Classifier_Name': m_name,
                'Compute_Engine': dev_used,
                'Acc_Mean_Pct': round(m_acc, 2),
                'Acc_SD_Pct': round(s_acc, 2),
                'Accuracy_Formatted': f"{m_acc:.2f}% ± {s_acc:.2f}%",
                'MacroF1_Mean_Pct': round(m_f1, 2),
                'MacroF1_SD_Pct': round(s_f1, 2),
                'MacroF1_Formatted': f"{m_f1:.2f}% ± {s_f1:.2f}%",
                'Macro_Sensitivity_Pct': round(m_rec, 2),
                'Macro_Precision_Pct': round(m_prec, 2),
                'Mean_Fold_ROC_AUC': round(m_auc, 4),
                'Global_OOF_ROC_AUC': round(float(g_auc), 4),
                'Best_Fold_Acc_Pct': round(float(np.max(fold_accs[m_name])), 2),
                'Total_Time_Sec': round(model_times[m_name], 3)
            })

            cost_rows.append({
                'Task_Scenario': t_name,
                'Classifier_Name': m_name,
                'Compute_Engine': dev_used,
                'Total_Segments': n_samples,
                'CV_Folds': OUTER_FOLDS,
                'Mean_k_Compact': round(float(np.mean(chosen_k_comp_list)), 1),
                'Mean_k_Rich': round(float(np.mean(chosen_k_rich_list)), 1),
                'Total_Time_Sec': round(model_times[m_name], 3),
                'Mean_Time_Per_Fold_Sec': round(model_times[m_name] / OUTER_FOLDS, 3),
                'Latency_ms_Per_Segment': round((model_times[m_name] * 1000.0) / n_samples, 3),
                'Peak_GPU_VRAM_MB': round(peak_vram_mb, 2),
                'Peak_System_RAM_MB': round(peak_ram_mb, 2)
            })

            print(f"    -> {m_name:<30} | Acc: {m_acc:6.2f}% ± {s_acc:4.2f}% | "
                  f"F1: {m_f1:6.2f}% | AUC: {g_auc:.3f}")

            if (0.65 * m_f1 + 35.0 * g_auc) > win_score:
                win_score = 0.65 * m_f1 + 35.0 * g_auc
                win_model = m_name

        best_model_winner[t_name] = win_model
        best_oof_preds[t_name]    = oof_preds[win_model]
        best_oof_probs[t_name]    = oof_probs[win_model]
        best_y_true[t_name]       = y_task

        w_acc_m = float(np.mean(fold_accs[win_model]))
        w_acc_s = float(np.std(fold_accs[win_model]))
        w_f1_m  = float(np.mean(fold_f1s[win_model]))
        w_f1_s  = float(np.std(fold_f1s[win_model]))
        if num_cls == 2:
            w_auc = float(roc_auc_score(y_task, oof_probs[win_model][:, 1]))
        else:
            w_auc = float(roc_auc_score(y_task, oof_probs[win_model], multi_class='ovr', average='macro'))

        pb = PREV_BENCHMARKS[t_name]
        exec_summary_rows.append({
            'Task_Scenario': t_name,
            'Winning_Step5_Model': win_model,
            'Step2_Acc_Pct': pb['S2_Acc'],
            'Step3_Acc_Pct': pb['S3_Acc'],
            'Step4_MATLAB_Fused_Acc_Pct': pb['S4_Acc'],
            'Step5_T4_Fused_Acc_Pct': round(w_acc_m, 2),
            'Step5_Acc_SD_Pct': round(w_acc_s, 2),
            'Step5_Accuracy_Formatted': f"{w_acc_m:.2f}% ± {w_acc_s:.2f}%",
            'Step5_MacroF1_Formatted': f"{w_f1_m:.2f}% ± {w_f1_s:.2f}%",
            'Step4_AUC': pb['S4_AUC'],
            'Step5_OOF_ROC_AUC': round(w_auc, 4),
            'Best_Fold_Acc_Pct': round(float(np.max(fold_accs[win_model])), 2)
        })

        cm_pdf = f"Confusion_Matrix_Step5_T4_{t_name}.pdf"
        save_clean_confusion_matrix_pdf(y_task, oof_preds[win_model], c_labels, cm_pdf)
        generated_files.append(cm_pdf)

    # =========================================================================
    # 10. GENERATE STANDALONE ROC PDF + 6-PANEL DASHBOARD (PNG + PDF)
    # =========================================================================
    print("\nGenerating Step 5 Vector PDFs and 6-Panel Summary Dashboard...")

    roc_pdf = "ROC_Curves_Step5_T4_Pairwise_Tasks.pdf"
    plt.figure(figsize=(4.6, 3.6))
    roc_colors = ['#1b9e77', '#d95f02', '#7570b3']
    for idx, task in enumerate(TASKS[:3]):
        t_name = task['name']
        yt = best_y_true[t_name]
        pr = best_oof_probs[t_name][:, 1]
        fpr, tpr, _ = roc_curve(yt, pr)
        auc_v = roc_auc_score(yt, pr)
        plt.plot(fpr, tpr, lw=1.6, color=roc_colors[idx],
                 label=f"{t_name.replace('_', ' ')} (AUC = {auc_v:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Random Chance (0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=8, fontweight='normal')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=8, fontweight='normal')
    plt.xticks(fontsize=8, fontweight='normal')
    plt.yticks(fontsize=8, fontweight='normal')
    plt.legend(loc='lower right', fontsize=7, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_pdf, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
    generated_files.append(roc_pdf)

    fig = plt.figure(figsize=(15.0, 9.2))
    gs  = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, task in enumerate(TASKS):
        t_name   = task['name']
        c_labels = task['labels']
        yt       = best_y_true[t_name]
        yp       = best_oof_preds[t_name]
        w_m      = best_model_winner[t_name]

        r_pos, c_pos = subplot_positions[idx]
        ax = fig.add_subplot(gs[r_pos, c_pos])

        cm = confusion_matrix(yt, yp)
        cm_perc = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0
        annot_mat = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_mat[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

        sns.heatmap(cm_perc, annot=annot_mat, fmt='', cmap='YlGnBu',
                    xticklabels=c_labels, yticklabels=c_labels, cbar=False,
                    ax=ax, annot_kws={'size': 8.5, 'weight': 'normal'})
        acc_v = accuracy_score(yt, yp) * 100.0
        ax.set_title(f"{t_name.replace('_', ' ')}\n({w_m}: {acc_v:.2f}%)", fontsize=9.5, fontweight='normal')
        ax.set_xlabel('Predicted Emotion', fontsize=8, fontweight='normal')
        ax.set_ylabel('True Emotion', fontsize=8, fontweight='normal')
        ax.tick_params(labelsize=8)

    ax_roc = fig.add_subplot(gs[0, 2])
    for idx, task in enumerate(TASKS[:3]):
        t_name = task['name']
        yt = best_y_true[t_name]
        pr = best_oof_probs[t_name][:, 1]
        fpr, tpr, _ = roc_curve(yt, pr)
        auc_v = roc_auc_score(yt, pr)
        ax_roc.plot(fpr, tpr, lw=1.8, color=roc_colors[idx],
                    label=f"{t_name.replace('_', ' ')} (AUC={auc_v:.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Chance (0.500)')
    ax_roc.set_title('Step 5 Fused Out-Of-Fold ROC Curves (T4 GPU)', fontsize=9.5, fontweight='normal')
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=8, fontweight='normal')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=8, fontweight='normal')
    ax_roc.tick_params(labelsize=8)
    ax_roc.legend(loc='lower right', fontsize=7.5)
    ax_roc.grid(True, linestyle='--', alpha=0.5)

    ax_bar = fig.add_subplot(gs[1, 2])
    df_exec_plot = pd.DataFrame(exec_summary_rows)
    x_idx = np.arange(len(TASKS))
    w_bar = 0.20

    ax_bar.bar(x_idx - 1.5*w_bar, df_exec_plot['Step2_Acc_Pct'], width=w_bar,
               color='#7570b3', edgecolor='black', label='Step 2 Acc (%)')
    ax_bar.bar(x_idx - 0.5*w_bar, df_exec_plot['Step4_MATLAB_Fused_Acc_Pct'], width=w_bar,
               color='#2b5c8f', edgecolor='black', label='Step 4 Acc (%)')
    ax_bar.bar(x_idx + 0.5*w_bar, df_exec_plot['Step5_T4_Fused_Acc_Pct'], width=w_bar,
               color='#1b9e77', edgecolor='black', label='Step 5 T4 Acc (%)')
    ax_bar.bar(x_idx + 1.5*w_bar, df_exec_plot['Step5_OOF_ROC_AUC'] * 100.0, width=w_bar,
               color='#d95f02', edgecolor='black', label='Step 5 AUC (%)')

    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(['Amu vs Con', 'Amu vs Fea', 'Con vs Fea', 'Triple (3-Cls)'], fontsize=8)
    ax_bar.set_ylim(45, 102)
    ax_bar.set_ylabel('Performance Score (%)', fontsize=8, fontweight='normal')
    ax_bar.set_title('Progression: Step 2 vs Step 4 vs Step 5 (T4 GPU)', fontsize=9.5, fontweight='normal')
    ax_bar.legend(loc='upper right', fontsize=7, ncol=2)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    dash_png = 'Step5_T4_GPU_Fused3570_Dashboard.png'
    dash_pdf = 'Step5_T4_GPU_Fused3570_Dashboard.pdf'
    plt.savefig(dash_png, dpi=300, bbox_inches='tight')
    plt.savefig(dash_pdf, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
    generated_files.extend([dash_png, dash_pdf])

    # =========================================================================
    # 11. EXPORT 6-SHEET EXCEL WORKBOOK & AUTO-DOWNLOAD ZIP
    # =========================================================================
    df_exec = pd.DataFrame(exec_summary_rows)
    df_all  = pd.DataFrame(all_models_rows)
    df_fold = pd.DataFrame(per_fold_rows)
    df_k    = pd.DataFrame(k_search_rows)
    df_feat = pd.DataFrame(selected_feat_rows)
    df_cost = pd.DataFrame(cost_rows)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_exec.to_excel(writer, sheet_name='Executive_Summary_S2_to_S5',  index=False)
        df_all.to_excel(writer,  sheet_name='All_Models_Step5_Benchmark',  index=False)
        df_fold.to_excel(writer, sheet_name='Per_Fold_Detailed_Metrics',   index=False)
        df_k.to_excel(writer,    sheet_name='Optimal_k_Search_Analysis',   index=False)
        df_feat.to_excel(writer, sheet_name='Top_Selected_Fused_Features', index=False)
        df_cost.to_excel(writer, sheet_name='Computational_Cost_and_Time', index=False)

    generated_files.append(OUTPUT_EXCEL)

    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f_path in generated_files:
            if os.path.exists(f_path):
                zf.write(f_path)

    total_elapsed = time.time() - pipeline_t0
    print(f"\n======================================================================")
    print(f"STEP 5 T4 GPU FUSED-3570 PIPELINE COMPLETED IN {total_elapsed:.1f} SECONDS!")
    print(f"Excel Report : {OUTPUT_EXCEL}")
    print(f"Dashboard PNG: {dash_png}")
    print(f"ZIP Archive  : {ZIP_FILENAME}")
    print(f"======================================================================\n")
    print(df_exec[['Task_Scenario', 'Winning_Step5_Model', 'Step4_MATLAB_Fused_Acc_Pct',
                   'Step5_Accuracy_Formatted', 'Step5_OOF_ROC_AUC', 'Best_Fold_Acc_Pct']].to_string(index=False))

    try:
        from google.colab import files
        files.download(ZIP_FILENAME)
    except Exception:
        print("Files saved locally in current directory.")


if __name__ == '__main__':
    main()