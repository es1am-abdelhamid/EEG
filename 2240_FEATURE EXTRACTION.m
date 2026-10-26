%% =========================================================================
% COMPREHENSIVE 2240-FEATURE EXTRACTION (TIME + FREQ + WAVELET DB4 & COIF5),
% MULTI-TEST STATISTICAL RANKING, AND AUTOMATED DIAGNOSTIC FIGURE GENERATION
% Dataset:  Amusement (4), Confusion (11), Fear (17)
% Per Channel (160 Features):
%   - 40 Time-Domain & Non-Linear Dynamical Features (T01..T40)
%   - 40 Frequency-Domain Spectral & Dynamic Features (F01..F40)
%   - 40 Wavelet-Domain Features using 'db4' (DB4_W01..W40)
%   - 40 Wavelet-Domain Features using 'coif5' (COIF5_W01..W40)
% Total Across 14 Channels = 14 x 160 = 2240 Features per Segment
% =========================================================================
clear; clc; close all;
warning('off', 'all');

tic;
fprintf('=========================================================================\n');
fprintf('STARTING 2240-FEATURE EXTRACTION (TIME + FREQ + WAVELET) FOR (4, 11, 17)\n');
fprintf('=========================================================================\n');

%% 1. LOAD BALANCED EEG DATA (0.5 - 45 HZ)
filename_in = 'eeg_41711.mat';
if ~isfile(filename_in)
    error('File "%s" not found! Please ensure it is in the current directory.', filename_in);
end

fprintf('Loading "%s"...\n', filename_in);
load(filename_in); % Expected: Segments_Balanced, EmotionID_Balanced, ParticipantID_Balanced, Fs

EEG_Data        = Segments_Balanced;
EmotionID_3     = double(EmotionID_Balanced(:));
ParticipantID_3 = double(ParticipantID_Balanced(:));
target_emotions = [4, 11, 17];
emotion_names   = {'Amusement', 'Confusion', 'Fear'};

[seq_len, num_ch, num_seg] = size(EEG_Data);
fprintf('Dataset Ready: %d Segments | %d Channels | %d Time Samples/Segment.\n', ...
    num_seg, num_ch, seq_len);

%% 2. DEFINE THE 160 FEATURE NAMES PER CHANNEL (40 TIME + 40 FREQ + 40 DB4 + 40 COIF5)
time_feat_names = { ...
    'T01_Variance_Activity', 'T02_StdDev', 'T03_RMS', 'T04_MeanAbsValue', 'T05_Skewness', ...
    'T06_Kurtosis', 'T07_IQR', 'T08_MeanAbsDev', 'T09_MedianAbsDev', 'T10_PeakToPeak', ...
    'T11_MaxAbsAmp', 'T12_CrestFactor', 'T13_ShapeFactor', 'T14_ImpulseFactor', 'T15_ClearanceFactor', ...
    'T16_Hjorth_Mobility', 'T17_Hjorth_Complexity', 'T18_ZeroCrossRate', 'T19_SlopeSignChanges', 'T20_SecondDiff_ZCR', ...
    'T21_WaveformLength', 'T22_NormFirstDiff', 'T23_MeanAbsSecondDiff', 'T24_NormSecondDiff', 'T25_TKEO_Mean', ...
    'T26_TKEO_Std', 'T27_TKEO_Skewness', 'T28_TKEO_Kurtosis', 'T29_LogEnergy_Entropy', 'T30_ShannonEnergy_Entropy', ...
    'T31_PermEntropy_Lag1', 'T32_PermEntropy_Lag2', 'T33_Petrosian_FD', 'T34_Katz_FD', 'T35_Higuchi_FD', ...
    'T36_Autocorr_Lag1', 'T37_Autocorr_Lag2', 'T38_Autocorr_Lag4', 'T39_NonStat_Var_CV', 'T40_NonStat_MAV_CV'};

freq_feat_names = { ...
    'F01_Delta_RelPow', 'F02_Theta_RelPow', 'F03_Alpha_RelPow', 'F04_Beta_RelPow', 'F05_Gamma_RelPow', ...
    'F06_Delta_AbsPow', 'F07_Theta_AbsPow', 'F08_Alpha_AbsPow', 'F09_Beta_AbsPow', 'F10_Gamma_AbsPow', ...
    'F11_Log_DeltaPow', 'F12_Log_ThetaPow', 'F13_Log_AlphaPow', 'F14_Log_BetaPow', 'F15_Log_GammaPow', ...
    'F16_Ratio_DeltaTheta', 'F17_Ratio_ThetaAlpha', 'F18_Ratio_AlphaBeta', 'F19_Ratio_BetaGamma', 'F20_Ratio_ThetaBeta', ...
    'F21_Spec_Centroid', 'F22_Spec_Spread', 'F23_Spec_Skewness', 'F24_Spec_Kurtosis', 'F25_Spec_Entropy', ...
    'F26_Spec_Flatness', 'F27_Spec_Rolloff85', 'F28_Spec_Bandwidth', 'F29_Peak_Frequency', 'F30_Peak_Power', ...
    'F31_Median_Frequency', 'F32_Mean_Frequency', 'F33_Spec_Slope', 'F34_Spec_Decrease', 'F35_Spec_Irregularity', ...
    'F36_Spec_CrestFactor', 'F37_Spec_EntropyRate', 'F38_Spec_Flux', 'F39_Spec_Moment1', 'F40_Spec_Moment2'};

wav_base_names = { ...
    'W01_Eng_A5', 'W02_Eng_D1', 'W03_Eng_D2', 'W04_Eng_D3', 'W05_Eng_D4', 'W06_Eng_D5', ...
    'W07_EngRatio_A5', 'W08_EngRatio_D1', 'W09_EngRatio_D2', 'W10_EngRatio_D3', 'W11_EngRatio_D4', 'W12_EngRatio_D5', ...
    'W13_ShanEnt_A5', 'W14_ShanEnt_D1', 'W15_ShanEnt_D2', 'W16_ShanEnt_D3', 'W17_ShanEnt_D4', 'W18_ShanEnt_D5', ...
    'W19_RMS_A5', 'W20_RMS_D1', 'W21_RMS_D2', 'W22_RMS_D3', 'W23_RMS_D4', 'W24_RMS_D5', ...
    'W25_Kurt_A5', 'W26_Kurt_D1', 'W27_Kurt_D2', 'W28_Kurt_D3', 'W29_Kurt_D4', 'W30_Kurt_D5', ...
    'W31_Skew_A5', 'W32_Skew_D1', 'W33_Skew_D2', 'W34_Skew_D3', 'W35_Skew_D4', 'W36_Skew_D5', ...
    'W37_LogEng_Entropy', 'W38_Coeff_Sparsity', 'W39_Coeff_ZeroCrossRate', 'W40_Detail_EngDist_Entropy'};

db4_feat_names   = cellfun(@(s) ['DB4_' s],   wav_base_names, 'UniformOutput', false);
coif5_feat_names = cellfun(@(s) ['COIF5_' s], wav_base_names, 'UniformOutput', false);

single_ch_feat_names = [time_feat_names, freq_feat_names, db4_feat_names, coif5_feat_names];
num_feat_per_ch = length(single_ch_feat_names); % 160 features per channel
total_features  = num_ch * num_feat_per_ch;      % 14 * 160 = 2240 features

Feature_Names_2240   = cell(total_features, 1);
Channel_Per_Feature  = cell(total_features, 1);
Domain_Per_Feature   = cell(total_features, 1);
BaseName_Per_Feature = cell(total_features, 1);

idx_ptr = 1;
for ch = 1:num_ch
    ch_str = sprintf('Channel_%02d', ch);
    for f_i = 1:num_feat_per_ch
        b_name = single_ch_feat_names{f_i};
        Feature_Names_2240{idx_ptr}   = sprintf('%s_%s', ch_str, b_name);
        Channel_Per_Feature{idx_ptr}  = ch_str;
        BaseName_Per_Feature{idx_ptr} = b_name;
        if f_i <= 40
            Domain_Per_Feature{idx_ptr} = 'Time_Domain';
        elseif f_i <= 80
            Domain_Per_Feature{idx_ptr} = 'Frequency_Domain';
        elseif f_i <= 120
            Domain_Per_Feature{idx_ptr} = 'Wavelet_db4';
        else
            Domain_Per_Feature{idx_ptr} = 'Wavelet_coif5';
        end
        idx_ptr = idx_ptr + 1;
    end
end

%% 3. PRECOMPUTE SPECTRAL GRIDS
nfft = max(256, 2^nextpow2(seq_len));
win_len = min(256, seq_len);
win_hamming = hamming(win_len);
noverlap = floor(win_len / 2);
[~, f_grid] = pwelch(double(EEG_Data(:, 1, 1)), win_hamming, noverlap, nfft, Fs);
df = f_grid(2) - f_grid(1);

idx_band  = (f_grid >= 0.5 & f_grid <= 45);
f_sub     = f_grid(idx_band);
K_bins    = length(f_sub);

idx_delta = (f_grid >= 0.5 & f_grid < 4);
idx_theta = (f_grid >= 4   & f_grid < 8);
idx_alpha = (f_grid >= 8   & f_grid < 13);
idx_beta  = (f_grid >= 13  & f_grid < 30);
idx_gamma = (f_grid >= 30  & f_grid<= 45);

st_win_len = min(128, floor(seq_len / 2));
st_step    = max(1, floor(st_win_len / 2));
st_starts  = 1:st_step:(seq_len - st_win_len + 1);
num_st_win = length(st_starts);
st_hamming = hamming(st_win_len);
f_st       = (0:(st_win_len/2))' * (Fs / st_win_len);
idx_st_band = (f_st >= 0.5 & f_st <= 45);

%% 4. EXTRACT ALL 2240 FEATURES ACROSS ALL SEGMENTS
Features_2240 = zeros(num_seg, total_features);
fprintf('\nExtracting 160 features x 14 channels = 2240 features for %d segments...\n', num_seg);

for s = 1:num_seg
    row_feats = zeros(1, total_features);
    
    for ch = 1:num_ch
        sig = double(EEG_Data(:, ch, s));
        sig = sig(:) - mean(sig); % Ensure column vector & zero-mean
        
        % A) 40 TIME-DOMAIN & NON-LINEAR DYNAMICAL FEATURES
        t_vec_40 = extract_40_time_features(sig);
        
        % B) 40 FREQUENCY-DOMAIN FEATURES
        [pxx, ~] = pwelch(sig, win_hamming, noverlap, nfft, Fs);
        
        P_delta = sum(pxx(idx_delta)) * df;
        P_theta = sum(pxx(idx_theta)) * df;
        P_alpha = sum(pxx(idx_alpha)) * df;
        P_beta  = sum(pxx(idx_beta))  * df;
        P_gamma = sum(pxx(idx_gamma)) * df;
        P_total = P_delta + P_theta + P_alpha + P_beta + P_gamma + eps;
        
        Rel_delta = P_delta / P_total; Rel_theta = P_theta / P_total;
        Rel_alpha = P_alpha / P_total; Rel_beta  = P_beta  / P_total;
        Rel_gamma = P_gamma / P_total;
        
        Log_delta = log10(P_delta + eps); Log_theta = log10(P_theta + eps);
        Log_alpha = log10(P_alpha + eps); Log_beta  = log10(P_beta  + eps);
        Log_gamma = log10(P_gamma + eps);
        
        Ratio_DT = P_delta / (P_theta + eps); Ratio_TA = P_theta / (P_alpha + eps);
        Ratio_AB = P_alpha / (P_beta  + eps); Ratio_BG = P_beta  / (P_gamma + eps);
        Ratio_TB = P_theta / (P_beta  + eps);
        
        P_sub = pxx(idx_band) + eps;
        p_norm = P_sub / sum(P_sub);
        
        spec_centroid = sum(f_sub .* p_norm);
        spec_spread = sqrt(sum(((f_sub - spec_centroid) .^ 2) .* p_norm) + eps);
        spec_skew = sum(((f_sub - spec_centroid) .^ 3) .* p_norm) / ((spec_spread ^ 3) + eps);
        spec_kurt = sum(((f_sub - spec_centroid) .^ 4) .* p_norm) / ((spec_spread ^ 4) + eps);
        spec_ent = -sum(p_norm .* log2(p_norm + eps)) / log2(K_bins);
        spec_flatness = exp(mean(log(P_sub))) / (mean(P_sub) + eps);
        
        cum_p = cumsum(p_norm);
        idx_r85 = find(cum_p >= 0.85, 1, 'first'); if isempty(idx_r85), idx_r85 = K_bins; end
        spec_rolloff = f_sub(idx_r85);
        
        idx_q25 = find(cum_p >= 0.25, 1, 'first'); if isempty(idx_q25), idx_q25 = 1; end
        idx_q75 = find(cum_p >= 0.75, 1, 'first'); if isempty(idx_q75), idx_q75 = K_bins; end
        spec_bw = f_sub(idx_q75) - f_sub(idx_q25);
        
        [peak_pow, idx_pk] = max(P_sub); peak_freq = f_sub(idx_pk);
        
        idx_med = find(cum_p >= 0.50, 1, 'first'); if isempty(idx_med), idx_med = round(K_bins / 2); end
        median_freq = f_sub(idx_med);
        
        amp_sub = sqrt(P_sub);
        mean_freq = sum(f_sub .* amp_sub) / (sum(amp_sub) + eps);
        
        log_f = log10(f_sub); log_P = log10(P_sub);
        spec_slope = mean((log_f - mean(log_f)) .* (log_P - mean(log_P))) / (mean((log_f - mean(log_f)) .^ 2) + eps);
        
        k_vec = (1:(K_bins - 1))';
        spec_decrease = sum((P_sub(2:end) - P_sub(1)) ./ k_vec) / (sum(P_sub(2:end)) + eps);
        spec_irreg = sum((P_sub(1:end-1) - P_sub(2:end)) .^ 2) / (sum(P_sub .^ 2) + eps);
        spec_crest = peak_pow / (mean(P_sub) + eps);
        
        if num_st_win >= 2
            st_mat = zeros(st_win_len, num_st_win);
            for w_i = 1:num_st_win
                st_mat(:, w_i) = sig(st_starts(w_i):(st_starts(w_i) + st_win_len - 1)) .* st_hamming;
            end
            Y_st = abs(fft(st_mat, st_win_len, 1));
            S_st = (Y_st(idx_st_band, :) .^ 2) + eps;
            P_st_norm = bsxfun(@rdivide, S_st, sum(S_st, 1) + eps);
            
            st_entropies = -sum(P_st_norm .* log2(P_st_norm + eps), 1) / log2(size(P_st_norm, 1));
            spec_ent_rate = mean(abs(diff(st_entropies)));
            
            M_st_norm = bsxfun(@rdivide, sqrt(S_st), sqrt(sum(S_st, 1)) + eps);
            spec_flux = mean(sqrt(sum(diff(M_st_norm, 1, 2) .^ 2, 1)));
        else
            spec_ent_rate = 0; spec_flux = 0;
        end
        
        spec_mom1 = mean(f_sub .* P_sub);
        spec_mom2 = mean((f_sub .^ 2) .* P_sub);
        
        f_vec_40 = [Rel_delta, Rel_theta, Rel_alpha, Rel_beta, Rel_gamma, ...
                    P_delta, P_theta, P_alpha, P_beta, P_gamma, ...
                    Log_delta, Log_theta, Log_alpha, Log_beta, Log_gamma, ...
                    Ratio_DT, Ratio_TA, Ratio_AB, Ratio_BG, Ratio_TB, ...
                    spec_centroid, spec_spread, spec_skew, spec_kurt, spec_ent, ...
                    spec_flatness, spec_rolloff, spec_bw, peak_freq, peak_pow, ...
                    median_freq, mean_freq, spec_slope, spec_decrease, spec_irreg, ...
                    spec_crest, spec_ent_rate, spec_flux, spec_mom1, spec_mom2];
        
        % C) 80 WAVELET FEATURES (40 'db4' + 40 'coif5')
        w_db4_40   = extract_40_wavelet_features(sig, 'db4',   5);
        w_coif5_40 = extract_40_wavelet_features(sig, 'coif5', 5);
        
        c_start = (ch - 1) * num_feat_per_ch + 1;
        c_end   = ch * num_feat_per_ch;
        row_feats(c_start:c_end) = [t_vec_40, f_vec_40, w_db4_40, w_coif5_40];
    end
    
    Features_2240(s, :) = row_feats;
    if mod(s, 100) == 0 || s == num_seg
        fprintf('  Processed %d / %d segments (%.1f%%)...\n', s, num_seg, (s/num_seg)*100);
    end
end

Features_2240(isnan(Features_2240) | isinf(Features_2240)) = 0;

mat_out_file = 'eeg_2240features_AmuConFea.mat';
fprintf('\nSaving full 2240-feature matrix to "%s"...\n', mat_out_file);
save(mat_out_file, 'Features_2240', 'Feature_Names_2240', 'Channel_Per_Feature', ...
     'Domain_Per_Feature', 'BaseName_Per_Feature', 'EmotionID_3', ...
     'ParticipantID_3', 'target_emotions', 'emotion_names', '-v7.3');

%% 5. STATISTICAL EVALUATION ENGINE ACROSS PAIRS & TRIPLE
fprintf('\nRunning Multi-Test Statistical Evaluation across all 2240 features...\n');

pairs_cfg = {
    4,  11, 'Amusement', 'Confusion', 'Amu_vs_Con';
    4,  17, 'Amusement', 'Fear',      'Amu_vs_Fea';
    11, 17, 'Confusion', 'Fear',      'Con_vs_Fea'
};

TOP_N = 100;

file_auc     = 'Test1_Top100_AUC_Analysis.xlsx';
file_pearson = 'Test2_Top100_Pearson_Correlation.xlsx';
file_anova   = 'Test3_Top100_ANOVA_Fisher_FTest.xlsx';
file_mw_kw   = 'Test4_Top100_MannWhitney_Kruskal.xlsx';
file_ks      = 'Test5_Top100_KS_Distribution_Test.xlsx';
file_master  = 'Master_Summary_Top100_All_Tests.xlsx';

out_files = {file_auc, file_pearson, file_anova, file_mw_kw, file_ks, file_master};
for f_i = 1:length(out_files)
    if isfile(out_files{f_i}), delete(out_files{f_i}); end
end

AllTasks_MaxAUC   = zeros(total_features, 4);
AllTasks_AbsCorr  = zeros(total_features, 4);
AllTasks_AnovaF   = zeros(total_features, 4);
AllTasks_MW_KW_Z  = zeros(total_features, 4);
AllTasks_KS_Stat  = zeros(total_features, 4);

% 5A. PAIRWISE EVALUATION
for p_idx = 1:size(pairs_cfg, 1)
    id1   = pairs_cfg{p_idx, 1};
    id2   = pairs_cfg{p_idx, 2};
    name1 = pairs_cfg{p_idx, 3};
    name2 = pairs_cfg{p_idx, 4};
    s_tag = pairs_cfg{p_idx, 5};
    
    fprintf('  -> Pair %d/3: %s vs %s...\n', p_idx, name1, name2);
    
    mask1 = (EmotionID_3 == id1); mask2 = (EmotionID_3 == id2);
    mask_pair = mask1 | mask2;
    
    X_pair = Features_2240(mask_pair, :);
    y_pair = double(EmotionID_3(mask_pair) == id2);
    
    X1 = Features_2240(mask1, :); X2 = Features_2240(mask2, :);
    n1 = size(X1, 1); n2 = size(X2, 1);
    
    Mean1 = mean(X1, 1)'; SD1 = std(X1, 0, 1)';
    Mean2 = mean(X2, 1)'; SD2 = std(X2, 0, 1)';
    
    Raw_AUC = zeros(total_features, 1); Max_AUC = zeros(total_features, 1);
    MW_PVal = ones(total_features, 1);  MW_ZScore = zeros(total_features, 1);
    Pearson_r = zeros(total_features, 1); Abs_Pearson = zeros(total_features, 1); Pearson_PVal = ones(total_features, 1);
    ANOVA_F = zeros(total_features, 1); ANOVA_PVal = ones(total_features, 1); Eta_Sq = zeros(total_features, 1);
    KS_Stat = zeros(total_features, 1); KS_PVal = ones(total_features, 1);
    
    for f_i = 1:total_features
        col_all = X_pair(:, f_i); c1 = X1(:, f_i); c2 = X2(:, f_i);
        if std(col_all) < 1e-12, Raw_AUC(f_i)=0.5; Max_AUC(f_i)=0.5; continue; end
        
        % 1. Exact Rank-Based AUC
        r_tied = tiedrank(col_all);
        R2 = sum(r_tied(y_pair == 1));
        u_stat = R2 - (n2 * (n2 + 1)) / 2.0;
        auc_val = u_stat / (n1 * n2);
        Raw_AUC(f_i) = auc_val; Max_AUC(f_i) = max(auc_val, 1.0 - auc_val);
        
        % 2. Mann-Whitney U
        [p_mw, ~, stats_mw] = ranksum(c1, c2);
        MW_PVal(f_i) = p_mw;
        if isfield(stats_mw, 'zval'), MW_ZScore(f_i) = abs(stats_mw.zval);
        else, MW_ZScore(f_i) = abs(norminv(max(min(p_mw / 2, 0.499999), 1e-300))); end
        
        % 3. Pearson Correlation
        [r_mat, p_mat] = corrcoef(col_all, y_pair);
        Pearson_r(f_i) = r_mat(1, 2); Abs_Pearson(f_i) = abs(r_mat(1, 2)); Pearson_PVal(f_i) = p_mat(1, 2);
        
        % 4. One-Way ANOVA F
        grand_mean = mean(col_all);
        ss_between = n1 * ((Mean1(f_i) - grand_mean)^2) + n2 * ((Mean2(f_i) - grand_mean)^2);
        ss_within  = sum((c1 - Mean1(f_i)).^2) + sum((c2 - Mean2(f_i)).^2) + eps;
        f_stat     = (ss_between / 1) / (ss_within / (n1 + n2 - 2));
        ANOVA_F(f_i) = f_stat; ANOVA_PVal(f_i) = 1.0 - fcdf(f_stat, 1, n1 + n2 - 2);
        Eta_Sq(f_i)  = ss_between / (ss_between + ss_within + eps);
        
        % 5. Two-Sample Kolmogorov-Smirnov (KS) Test
        [~, p_ks, ks_st] = kstest2(c1, c2);
        KS_Stat(f_i) = ks_st; KS_PVal(f_i) = p_ks;
    end
    
    AllTasks_MaxAUC(:, p_idx)  = Max_AUC;
    AllTasks_AbsCorr(:, p_idx) = Abs_Pearson;
    AllTasks_AnovaF(:, p_idx)  = ANOVA_F;
    AllTasks_MW_KW_Z(:, p_idx) = MW_ZScore;
    AllTasks_KS_Stat(:, p_idx) = KS_Stat;
    
    Consensus_Pair = 0.25*norm_01(Max_AUC) + 0.20*norm_01(Abs_Pearson) + 0.20*norm_01(ANOVA_F) + 0.20*norm_01(MW_ZScore) + 0.15*norm_01(KS_Stat);
                 
    T_Pair = table(Feature_Names_2240, Channel_Per_Feature, Domain_Per_Feature, BaseName_Per_Feature, ...
        Mean1, SD1, Mean2, SD2, Raw_AUC, Max_AUC, Pearson_r, Abs_Pearson, Pearson_PVal, ...
        ANOVA_F, Eta_Sq, ANOVA_PVal, MW_ZScore, MW_PVal, KS_Stat, KS_PVal, Consensus_Pair, ...
        'VariableNames', {'Full_Feature_Name', 'Channel', 'Domain', 'Feature_BaseName', ...
        [name1 '_Mean'], [name1 '_SD'], [name2 '_Mean'], [name2 '_SD'], ...
        'Raw_AUC', 'Discriminative_Max_AUC', 'Pearson_r', 'Abs_Pearson_r', 'Pearson_P_Value', ...
        'ANOVA_F_Score', 'Eta_Squared_Effect', 'ANOVA_P_Value', ...
        'MannWhitney_Abs_Z', 'MannWhitney_P_Value', 'KS_Statistic', 'KS_P_Value', 'Consensus_Score'});
    
    write_top_n_sheet(T_Pair, 'Discriminative_Max_AUC', TOP_N, file_auc,     s_tag);
    write_top_n_sheet(T_Pair, 'Abs_Pearson_r',          TOP_N, file_pearson, s_tag);
    write_top_n_sheet(T_Pair, 'ANOVA_F_Score',          TOP_N, file_anova,   s_tag);
    write_top_n_sheet(T_Pair, 'MannWhitney_Abs_Z',      TOP_N, file_mw_kw,   s_tag);
    write_top_n_sheet(T_Pair, 'KS_Statistic',           TOP_N, file_ks,      s_tag);
    write_top_n_sheet(T_Pair, 'Consensus_Score',        TOP_N, file_master,  ['Top100_' s_tag]);
end

% 5B. TRIPLE (3-CLASS) EVALUATION
fprintf('  -> Triple Task (Amusement vs Confusion vs Fear)...\n');
mask_c1 = (EmotionID_3 == 4); mask_c2 = (EmotionID_3 == 11); mask_c3 = (EmotionID_3 == 17);
X_c1 = Features_2240(mask_c1, :); n_c1 = size(X_c1, 1);
X_c2 = Features_2240(mask_c2, :); n_c2 = size(X_c2, 1);
X_c3 = Features_2240(mask_c3, :); n_c3 = size(X_c3, 1);

Mean_Amu = mean(X_c1, 1)'; SD_Amu = std(X_c1, 0, 1)';
Mean_Con = mean(X_c2, 1)'; SD_Con = std(X_c2, 0, 1)';
Mean_Fea = mean(X_c3, 1)'; SD_Fea = std(X_c3, 0, 1)';

Triple_Mean_OvR_AUC = mean(AllTasks_MaxAUC(:, 1:3), 2);
Triple_Max_Pair_AUC = max(AllTasks_MaxAUC(:, 1:3), [], 2);
Triple_Mean_AbsCorr = mean(AllTasks_AbsCorr(:, 1:3), 2);
Triple_Max_KS       = max(AllTasks_KS_Stat(:, 1:3), [], 2);

Triple_ANOVA_F = zeros(total_features, 1); Triple_ANOVA_PVal = ones(total_features, 1);
Triple_Eta_Sq = zeros(total_features, 1); Triple_KW_Chi2 = zeros(total_features, 1); Triple_KW_PVal = ones(total_features, 1);

N_all = num_seg;
for f_i = 1:total_features
    col_f = Features_2240(:, f_i);
    if std(col_f) < 1e-12, continue; end
    
    g_mean = mean(col_f);
    ss_b = n_c1*((Mean_Amu(f_i)-g_mean)^2) + n_c2*((Mean_Con(f_i)-g_mean)^2) + n_c3*((Mean_Fea(f_i)-g_mean)^2);
    ss_w = sum((X_c1(:, f_i)-Mean_Amu(f_i)).^2) + sum((X_c2(:, f_i)-Mean_Con(f_i)).^2) + sum((X_c3(:, f_i)-Mean_Fea(f_i)).^2) + eps;
    f_val = (ss_b / 2.0) / (ss_w / (N_all - 3));
    Triple_ANOVA_F(f_i) = f_val; Triple_ANOVA_PVal(f_i) = 1.0 - fcdf(f_val, 2, N_all - 3);
    Triple_Eta_Sq(f_i) = ss_b / (ss_b + ss_w + eps);
    
    r_all = tiedrank(col_f);
    H_kw = (12.0 / (N_all*(N_all+1))) * ((sum(r_all(mask_c1))^2)/n_c1 + (sum(r_all(mask_c2))^2)/n_c2 + (sum(r_all(mask_c3))^2)/n_c3) - 3.0*(N_all+1);
    Triple_KW_Chi2(f_i) = max(0, H_kw); Triple_KW_PVal(f_i) = 1.0 - chi2cdf(Triple_KW_Chi2(f_i), 2);
end

AllTasks_MaxAUC(:, 4)  = Triple_Mean_OvR_AUC; AllTasks_AbsCorr(:, 4) = Triple_Mean_AbsCorr;
AllTasks_AnovaF(:, 4)  = Triple_ANOVA_F; AllTasks_MW_KW_Z(:, 4) = Triple_KW_Chi2; AllTasks_KS_Stat(:, 4) = Triple_Max_KS;

Consensus_Triple = 0.25*norm_01(Triple_Mean_OvR_AUC) + 0.20*norm_01(Triple_Mean_AbsCorr) + 0.20*norm_01(Triple_ANOVA_F) + 0.20*norm_01(Triple_KW_Chi2) + 0.15*norm_01(Triple_Max_KS);

T_Triple = table(Feature_Names_2240, Channel_Per_Feature, Domain_Per_Feature, BaseName_Per_Feature, ...
    Mean_Amu, SD_Amu, Mean_Con, SD_Con, Mean_Fea, SD_Fea, ...
    Triple_Mean_OvR_AUC, Triple_Max_Pair_AUC, Triple_Mean_AbsCorr, ...
    Triple_ANOVA_F, Triple_Eta_Sq, Triple_ANOVA_PVal, Triple_KW_Chi2, Triple_KW_PVal, Triple_Max_KS, Consensus_Triple, ...
    'VariableNames', {'Full_Feature_Name', 'Channel', 'Domain', 'Feature_BaseName', ...
    'Amusement_Mean', 'Amusement_SD', 'Confusion_Mean', 'Confusion_SD', 'Fear_Mean', 'Fear_SD', ...
    'Mean_Pairwise_Max_AUC', 'Best_Pairwise_Max_AUC', 'Mean_Abs_Pearson_r', ...
    'ANOVA_3Class_F_Score', 'Eta_Squared_3Class', 'ANOVA_3Class_P_Value', ...
    'KruskalWallis_Chi2', 'KruskalWallis_P_Value', 'Max_Pairwise_KS_Stat', 'Consensus_Score'});

write_top_n_sheet(T_Triple, 'Mean_Pairwise_Max_AUC', TOP_N, file_auc,     'Triple_All_3');
write_top_n_sheet(T_Triple, 'Mean_Abs_Pearson_r',    TOP_N, file_pearson, 'Triple_All_3');
write_top_n_sheet(T_Triple, 'ANOVA_3Class_F_Score',  TOP_N, file_anova,   'Triple_All_3');
write_top_n_sheet(T_Triple, 'KruskalWallis_Chi2',    TOP_N, file_mw_kw,   'Triple_All_3');
write_top_n_sheet(T_Triple, 'Max_Pairwise_KS_Stat',  TOP_N, file_ks,      'Triple_All_3');
write_top_n_sheet(T_Triple, 'Consensus_Score',       TOP_N, file_master,  'Top100_Triple_All_3');

%% 6. GLOBAL SUMMARY (MASTER SHEET)
fprintf('\nBuilding Global Summary Top-100 Sheets...\n');
Global_Mean_MaxAUC  = mean(AllTasks_MaxAUC(:, 1:3), 2);
Global_Best_MaxAUC  = max(AllTasks_MaxAUC(:, 1:3), [], 2);
Global_Mean_AbsCorr = mean(AllTasks_AbsCorr(:, 1:3), 2);
Global_Mean_AnovaF  = mean(AllTasks_AnovaF, 2);
Global_Mean_MW_Z    = mean(AllTasks_MW_KW_Z(:, 1:3), 2);
Global_Mean_KS      = mean(AllTasks_KS_Stat(:, 1:3), 2);

Global_Consensus_Score = 0.25*norm_01(Global_Mean_MaxAUC) + 0.15*norm_01(Global_Best_MaxAUC) + ...
                         0.15*norm_01(Global_Mean_AbsCorr) + 0.15*norm_01(Global_Mean_AnovaF) + ...
                         0.15*norm_01(Global_Mean_MW_Z) + 0.15*norm_01(Global_Mean_KS);

T_Global = table(Feature_Names_2240, Channel_Per_Feature, Domain_Per_Feature, BaseName_Per_Feature, ...
    AllTasks_MaxAUC(:, 1), AllTasks_MaxAUC(:, 2), AllTasks_MaxAUC(:, 3), ...
    Global_Mean_MaxAUC, Global_Best_MaxAUC, Global_Mean_AbsCorr, Global_Mean_AnovaF, Triple_ANOVA_PVal, ...
    Global_Mean_MW_Z, Triple_KW_PVal, Global_Mean_KS, Global_Consensus_Score, ...
    'VariableNames', {'Full_Feature_Name', 'Channel', 'Domain', 'Feature_BaseName', ...
    'MaxAUC_Amu_vs_Con', 'MaxAUC_Amu_vs_Fea', 'MaxAUC_Con_vs_Fea', ...
    'Global_Mean_MaxAUC', 'Global_Peak_MaxAUC', 'Global_Mean_Abs_Pearson', 'Global_Mean_ANOVA_F', 'Triple_ANOVA_P_Value', ...
    'Global_Mean_MannWhitney_Z', 'Triple_Kruskal_P_Value', 'Global_Mean_KS_Stat', 'Master_Consensus_Score'});

write_top_n_sheet(T_Global, 'Global_Mean_MaxAUC',        TOP_N, file_auc,     'Summary_Top100_AUC');
write_top_n_sheet(T_Global, 'Global_Mean_Abs_Pearson',   TOP_N, file_pearson, 'Summary_Top100_Pearson');
write_top_n_sheet(T_Global, 'Global_Mean_ANOVA_F',       TOP_N, file_anova,   'Summary_Top100_ANOVA');
write_top_n_sheet(T_Global, 'Global_Mean_MannWhitney_Z', TOP_N, file_mw_kw,   'Summary_Top100_MW_KW');
write_top_n_sheet(T_Global, 'Global_Mean_KS_Stat',       TOP_N, file_ks,      'Summary_Top100_KS');

T_Master_Top100 = write_top_n_sheet(T_Global, 'Master_Consensus_Score', TOP_N, file_master, 'Summary_Consensus_Top100');

[uniq_ch, ~, ic_ch] = unique(T_Master_Top100.Channel);
[ch_counts_sorted, idx_ch_s] = sort(accumarray(ic_ch, 1), 'descend');
T_Ch_Breakdown = table(uniq_ch(idx_ch_s), ch_counts_sorted, 'VariableNames', {'Channel', 'Count_In_Top100'});
writetable(T_Ch_Breakdown, file_master, 'Sheet', 'Channel_Breakdown');

[uniq_dom, ~, ic_dom] = unique(T_Master_Top100.Domain);
[dom_counts_sorted, idx_dom_s] = sort(accumarray(ic_dom, 1), 'descend');
T_Dom_Breakdown = table(uniq_dom(idx_dom_s), dom_counts_sorted, 'VariableNames', {'Feature_Domain', 'Count_In_Top100'});
writetable(T_Dom_Breakdown, file_master, 'Sheet', 'Domain_Breakdown');

%% 7. AUTOMATED GENERATION OF THE 4-PANEL DIAGNOSTIC FIGURE (PNG + PDF)
fprintf('\nGenerating 4-Panel Diagnostic Figure directly in MATLAB...\n');

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0.5, 0.5, 13.5, 9.5], 'Visible', 'on');

% --- Panel 1 (Top-Left): Channel Breakdown Horizontal Bar Plot ---
ax1 = subplot(2, 2, 1);
ch_labels = T_Ch_Breakdown.Channel;
ch_vals   = T_Ch_Breakdown.Count_In_Top100;
b1 = barh(ax1, flipud(ch_vals), 0.65, 'FaceColor', [0.17, 0.36, 0.56], 'EdgeColor', 'k');
set(ax1, 'YTick', 1:length(ch_labels), 'YTickLabel', flipud(strrep(ch_labels, '_', '\_')), ...
    'FontSize', 8, 'FontWeight', 'normal');
xlabel(ax1, 'Number of Features in Global Top 100', 'FontSize', 8, 'FontWeight', 'normal');
ylabel(ax1, 'EEG Channel', 'FontSize', 8, 'FontWeight', 'normal');
title(ax1, 'Top 100 Consensus Features by EEG Channel', 'FontSize', 9, 'FontWeight', 'normal');
xlim(ax1, [0, max(ch_vals) + 4]);
grid(ax1, 'on');
vals_flipped = flipud(ch_vals);
for i = 1:length(vals_flipped)
    text(ax1, vals_flipped(i) + 0.4, i, sprintf('%d%%', vals_flipped(i)), ...
        'FontSize', 8, 'FontWeight', 'normal', 'VerticalAlignment', 'middle');
end

% --- Panel 2 (Top-Right): Feature Domain Breakdown Bar Plot ---
ax2 = subplot(2, 2, 2);
dom_labels = T_Dom_Breakdown.Feature_Domain;
dom_vals   = T_Dom_Breakdown.Count_In_Top100;
b2 = bar(ax2, dom_vals, 0.55, 'FaceColor', 'flat', 'EdgeColor', 'k');
dom_colors = [0.17, 0.36, 0.56; 0.85, 0.37, 0.01; 0.46, 0.44, 0.70; 0.11, 0.62, 0.47];
for k_d = 1:length(dom_vals)
    b2.CData(k_d, :) = dom_colors(mod(k_d-1, size(dom_colors,1)) + 1, :);
end
set(ax2, 'XTick', 1:length(dom_labels), 'XTickLabel', strrep(dom_labels, '_', '\_'), ...
    'FontSize', 8, 'FontWeight', 'normal');
xlabel(ax2, 'Feature Extraction Domain', 'FontSize', 8, 'FontWeight', 'normal');
ylabel(ax2, 'Count in Global Top 100', 'FontSize', 8, 'FontWeight', 'normal');
title(ax2, 'Top 100 Consensus Features by Extraction Domain', 'FontSize', 9, 'FontWeight', 'normal');
ylim(ax2, [0, max(dom_vals) + 8]);
grid(ax2, 'on');
for i = 1:length(dom_vals)
    text(ax2, i, dom_vals(i) + 1.5, sprintf('%d (%d%%)', dom_vals(i), dom_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'normal');
end

% --- Panel 3 (Bottom-Left): Top 15 Univariate Feature AUCs per Emotion Pair ---
ax3 = subplot(2, 2, 3);
top_k_plot = 15;
auc_p1 = sort(AllTasks_MaxAUC(:, 1), 'descend'); auc_p1 = auc_p1(1:top_k_plot);
auc_p2 = sort(AllTasks_MaxAUC(:, 2), 'descend'); auc_p2 = auc_p2(1:top_k_plot);
auc_p3 = sort(AllTasks_MaxAUC(:, 3), 'descend'); auc_p3 = auc_p3(1:top_k_plot);
auc_mat_15 = [auc_p1, auc_p2, auc_p3];

boxplot(ax3, auc_mat_15, 'Labels', {'Amusement vs Confusion', 'Amusement vs Fear', 'Confusion vs Fear'}, ...
    'Widths', 0.42, 'Colors', 'k', 'Symbol', '');
hold(ax3, 'on');
rng(42);
pair_colors = [0.11, 0.62, 0.47; 0.85, 0.37, 0.01; 0.46, 0.44, 0.70];
for p_i = 1:3
    jitter_x = p_i + (rand(top_k_plot, 1) - 0.5) * 0.20;
    scatter(ax3, jitter_x, auc_mat_15(:, p_i), 26, pair_colors(p_i, :), 'filled', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.85);
end
yline_val = 0.574;
plot(ax3, [0.5, 3.5], [yline_val, yline_val], 'r--', 'LineWidth', 1.2);
text(ax3, 0.6, yline_val + 0.002, 'Old Triad Best AUC (0.574)', 'Color', 'r', 'FontSize', 7.5);
hold(ax3, 'off');
set(ax3, 'FontSize', 8, 'FontWeight', 'normal');
ylabel(ax3, 'Discriminative Max AUC', 'FontSize', 8, 'FontWeight', 'normal');
title(ax3, 'Top 15 Univariate Feature AUCs per Emotion Pair', 'FontSize', 9, 'FontWeight', 'normal');
grid(ax3, 'on');

% --- Panel 4 (Bottom-Right): Heatmap of Top 10 Master Consensus Features ---
ax4 = subplot(2, 2, 4);
top10_names = T_Master_Top100.Full_Feature_Name(1:10);
top10_aucs  = [T_Master_Top100.MaxAUC_Amu_vs_Con(1:10), ...
               T_Master_Top100.MaxAUC_Amu_vs_Fea(1:10), ...
               T_Master_Top100.MaxAUC_Con_vs_Fea(1:10)];

imagesc(ax4, top10_aucs);
colormap(ax4, parula(256));
cb = colorbar(ax4);
cb.Label.String = 'Max AUC';
cb.Label.FontSize = 8;
cb.Label.FontWeight = 'normal';
set(ax4, 'XTick', 1:3, 'XTickLabel', {'Amu vs Con', 'Amu vs Fea', 'Con vs Fea'}, ...
    'YTick', 1:10, 'YTickLabel', strrep(top10_names, '_', '\_'), ...
    'FontSize', 7.5, 'FontWeight', 'normal');
title(ax4, 'Complementary Behavior of Top 10 Consensus Features', 'FontSize', 9, 'FontWeight', 'normal');

% Annotate heatmap cells with exact AUC values
mid_auc = (min(top10_aucs(:)) + max(top10_aucs(:))) / 2;
for r_i = 1:10
    for c_i = 1:3
        val_rc = top10_aucs(r_i, c_i);
        txt_col = 'k';
        if val_rc < mid_auc, txt_col = 'w'; end
        text(ax4, c_i, r_i, sprintf('%.3f', val_rc), ...
            'HorizontalAlignment', 'center', 'Color', txt_col, 'FontSize', 8, 'FontWeight', 'normal');
    end
end

% Export high-res PNG and Vector PDF
fig_png = 'Master_Summary_Diagnostic_2240Features.png';
fig_pdf = 'Master_Summary_Diagnostic_2240Features.pdf';
exportgraphics(fig, fig_png, 'Resolution', 300);
exportgraphics(fig, fig_pdf, 'ContentType', 'vector');

elapsed_sec = toc;
fprintf('\n=========================================================================\n');
fprintf('ALL DONE IN %.2f SECONDS!\n', elapsed_sec);
fprintf('Top #1 Feature Overall: %s (Peak AUC = %.4f, Mean AUC = %.4f)\n', ...
    T_Master_Top100.Full_Feature_Name{1}, T_Master_Top100.Global_Peak_MaxAUC(1), ...
    T_Master_Top100.Global_Mean_MaxAUC(1));
fprintf('Saved Diagnostic Figures: "%s" & "%s"\n', fig_png, fig_pdf);
fprintf('=========================================================================\n');

%% =========================================================================
% HELPER FUNCTIONS: 40 TIME-DOMAIN & 40 WAVELET FEATURE EXTRACTORS
% =========================================================================
function time_40 = extract_40_time_features(sig)
    % Fast, vectorized extraction of 40 Time-Domain & Non-Linear features
    N = length(sig);
    abs_sig = abs(sig);
    sq_sig  = sig .^ 2;
    
    % 1..6: Basic & Higher-Order Moments
    t01_var  = var(sig);
    t02_std  = sqrt(t01_var + eps);
    t03_rms  = sqrt(mean(sq_sig) + eps);
    t04_mav  = mean(abs_sig);
    z_sig    = sig / t02_std;
    t05_skew = mean(z_sig .^ 3);
    t06_kurt = mean(z_sig .^ 4);
    
    % 7..11: Robust Dispersion & Extrema
    s_sorted = sort(sig);
    q25 = s_sorted(max(1, round(0.25 * N)));
    q50 = s_sorted(max(1, round(0.50 * N)));
    q75 = s_sorted(max(1, round(0.75 * N)));
    t07_iqr     = q75 - q25;
    t08_mad     = mean(abs(sig));
    t09_med_ad  = median(abs(sig - q50));
    t10_p2p     = s_sorted(end) - s_sorted(1);
    t11_max_abs = max(abs_sig);
    
    % 12..15: Wave Shape & Impulse Factors
    t12_crest     = t11_max_abs / (t03_rms + eps);
    t13_shape     = t03_rms / (t04_mav + eps);
    t14_impulse   = t11_max_abs / (t04_mav + eps);
    t15_clearance = t11_max_abs / ((mean(sqrt(abs_sig)) ^ 2) + eps);
    
    % 16..17: Hjorth Mobility & Complexity
    dx1 = diff(sig);
    dx2 = diff(dx1);
    var_dx1 = var(dx1);
    var_dx2 = var(dx2);
    t16_hjorth_mob  = sqrt(var_dx1 / (t01_var + eps));
    mob_dx1         = sqrt(var_dx2 / (var_dx1 + eps));
    t17_hjorth_comp = mob_dx1 / (t16_hjorth_mob + eps);
    
    % 18..20: Zero-Crossing & Turning Point Rates
    t18_zcr     = sum((sig(1:end-1) .* sig(2:end)) < 0) / (N - 1);
    t19_ssc     = sum((dx1(1:end-1) .* dx1(2:end)) < 0) / (N - 2);
    t20_zcr_dx2 = sum((dx2(1:end-1) .* dx2(2:end)) < 0) / (N - 3);
    
    % 21..24: Waveform Length & Normalized First/Second Differences
    t21_wl             = mean(abs(dx1));
    t22_norm_diff1     = t21_wl / (t02_std + eps);
    d2_step            = sig(3:end) - sig(1:end-2);
    t23_mean_abs_diff2 = mean(abs(d2_step));
    t24_norm_diff2     = t23_mean_abs_diff2 / (t02_std + eps);
    
    % 25..28: Teager-Kaiser Energy Operator (TKEO) Statistics
    tkeo = (sig(2:end-1) .^ 2) - (sig(1:end-2) .* sig(3:end));
    t25_tkeo_mean = mean(tkeo);
    t26_tkeo_std  = std(tkeo) + eps;
    z_tkeo        = (tkeo - t25_tkeo_mean) / t26_tkeo_std;
    t27_tkeo_skew = mean(z_tkeo .^ 3);
    t28_tkeo_kurt = mean(z_tkeo .^ 4);
    
    % 29..32: Log-Energy, Shannon Energy, & Fast Permutation Entropies (m=3)
    t29_log_eng_ent     = mean(log10(sq_sig + eps));
    p_eng               = sq_sig / (sum(sq_sig) + eps);
    t30_shannon_eng_ent = -sum(p_eng .* log2(p_eng + eps)) / log2(N);
    t31_perm_ent_tau1   = fast_perm_entropy_ord3(sig, 1);
    t32_perm_ent_tau2   = fast_perm_entropy_ord3(sig, 2);
    
    % 33..35: Fractal Dimensions (Petrosian, Katz, & Multi-Scale Higuchi)
    N_delta = sum((dx1(1:end-1) .* dx1(2:end)) < 0);
    t33_petrosian_fd = log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta + eps)));
    L_curve = sum(abs(dx1));
    d_max   = max(abs(sig - sig(1))) + eps;
    t34_katz_fd = log10(N) / (log10(d_max / (L_curve + eps)) + log10(N) + eps);
    
    k_list = [1, 2, 4, 8];
    Lk = zeros(1, 4);
    for idx_k = 1:4
        k_val = k_list(idx_k);
        lm_sum = 0;
        for m = 1:k_val
            sub_x = sig(m:k_val:end);
            n_max = length(sub_x) - 1;
            if n_max > 0
                norm_f = (N - 1) / (n_max * k_val);
                lm_sum = lm_sum + (sum(abs(diff(sub_x))) * norm_f) / k_val;
            end
        end
        Lk(idx_k) = (lm_sum / k_val) + eps;
    end
    x_k = log(1.0 ./ k_list);
    y_k = log(Lk);
    t35_higuchi_fd = mean((x_k - mean(x_k)) .* (y_k - mean(y_k))) / (mean((x_k - mean(x_k)) .^ 2) + eps);
    
    % 36..38: Autocorrelation at Lags 1, 2, and 4
    denom_ac    = sum(sq_sig) + eps;
    t36_ac_lag1 = sum(sig(1:end-1) .* sig(2:end)) / denom_ac;
    t37_ac_lag2 = sum(sig(1:end-2) .* sig(3:end)) / denom_ac;
    t38_ac_lag4 = sum(sig(1:end-4) .* sig(5:end)) / denom_ac;
    
    % 39..40: Non-Stationarity Variation Across 10 Sub-Windows
    n_sub = 10;
    sub_len = floor(N / n_sub);
    if sub_len >= 8
        sub_mat  = reshape(sig(1:(n_sub * sub_len)), sub_len, n_sub);
        sub_vars = var(sub_mat, 0, 1);
        sub_mavs = mean(abs(sub_mat), 1);
        t39_nonstat_var_cv = std(sub_vars) / (mean(sub_vars) + eps);
        t40_nonstat_mav_cv = std(sub_mavs) / (mean(sub_mavs) + eps);
    else
        t39_nonstat_var_cv = 0;
        t40_nonstat_mav_cv = 0;
    end
    
    time_40 = [ ...
        t01_var, t02_std, t03_rms, t04_mav, t05_skew, t06_kurt, ...
        t07_iqr, t08_mad, t09_med_ad, t10_p2p, t11_max_abs, ...
        t12_crest, t13_shape, t14_impulse, t15_clearance, ...
        t16_hjorth_mob, t17_hjorth_comp, t18_zcr, t19_ssc, t20_zcr_dx2, ...
        t21_wl, t22_norm_diff1, t23_mean_abs_diff2, t24_norm_diff2, ...
        t25_tkeo_mean, t26_tkeo_std, t27_tkeo_skew, t28_tkeo_kurt, ...
        t29_log_eng_ent, t30_shannon_eng_ent, t31_perm_ent_tau1, t32_perm_ent_tau2, ...
        t33_petrosian_fd, t34_katz_fd, t35_higuchi_fd, ...
        t36_ac_lag1, t37_ac_lag2, t38_ac_lag4, ...
        t39_nonstat_var_cv, t40_nonstat_mav_cv];
end

function pe = fast_perm_entropy_ord3(x, tau)
    % Fast O(N) Bandt-Pompe Permutation Entropy for order m = 3
    x0 = x(1:(end - 2*tau));
    x1 = x((1 + tau):(end - tau));
    x2 = x((1 + 2*tau):end);
    code = 1 + 4 * double(x0 < x1) + 2 * double(x1 < x2) + double(x0 < x2);
    counts = accumarray(code, 1, [8, 1]);
    counts = counts(counts > 0);
    probs  = counts / sum(counts);
    pe     = -sum(probs .* log2(probs + eps)) / log2(6.0);
end

function wav_40 = extract_40_wavelet_features(sig, wname, level)
    [C, L] = wavedec(sig, level, wname);
    A5 = appcoef(C, L, wname, level);
    subbands = {A5, detcoef(C,L,1), detcoef(C,L,2), detcoef(C,L,3), detcoef(C,L,4), detcoef(C,L,5)};
    eng = zeros(1,6); sh_ent = zeros(1,6); rms_v = zeros(1,6); kurt = zeros(1,6); skew = zeros(1,6);
    for k = 1:6
        sb = subbands{k}(:); sq = sb.^2; eng(k) = sum(sq);
        p_k = sq / (eng(k) + eps); sh_ent(k) = -sum(p_k .* log2(p_k + eps)) / log2(max(length(sb), 2));
        rms_v(k) = sqrt(mean(sq) + eps);
        z_sb = (sb - mean(sb)) / (std(sb) + eps);
        kurt(k) = mean(z_sb.^4); skew(k) = mean(z_sb.^3);
    end
    eng_ratio = eng / (sum(eng) + eps);
    log_eng_ent = sum(log10((C(:).^2) + eps)) / length(C);
    sqrt_N = sqrt(length(C)); coeff_sparsity = (sqrt_N - (sum(abs(C)) / (sqrt(sum(C.^2)) + eps))) / (sqrt_N - 1 + eps);
    zcr_wav = sum((C(1:end-1) .* C(2:end)) < 0) / (length(C) - 1);
    q_det = eng(2:6) / (sum(eng(2:6)) + eps); det_dist_ent = -sum(q_det .* log2(q_det + eps)) / log2(5);
    wav_40 = [eng, eng_ratio, sh_ent, rms_v, kurt, skew, log_eng_ent, coeff_sparsity, zcr_wav, det_dist_ent];
end

function v_norm = norm_01(v)
    v = double(v(:)); v(isnan(v) | isinf(v)) = 0;
    mn = min(v); mx = max(v);
    if (mx - mn) < 1e-12, v_norm = zeros(size(v)); else, v_norm = (v - mn) / (mx - mn); end
end

function T_top = write_top_n_sheet(T_in, sort_col, top_n, excel_file, sheet_name)
    T_sorted = sortrows(T_in, sort_col, 'descend');
    n_keep = min(top_n, height(T_sorted));
    T_top = addvars(T_sorted(1:n_keep, :), (1:n_keep)', 'Before', 1, 'NewVariableNames', 'Rank');
    writetable(T_top, excel_file, 'Sheet', sheet_name);
end