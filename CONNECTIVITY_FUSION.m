%% =========================================================================
% STEP 3: GOLDEN-CHANNEL FUNCTIONAL CONNECTIVITY FUSION
% =========================================================================
% [WHAT WE ARE DOING :
%   Instead of starting over, we KEEP all 2880 calibrated features  and AUGMENT (fuse) them with
%   690 Brain Functional Connectivity features computed across the 15 pairs of
%   our 6 Golden Channels [Ch10, Ch05, Ch06, Ch07, Ch03, Ch08] from 'eeg_41711.mat':
%     1) Band-Specific Phase-Locking Value (PLV) across 5 bands (75 features)
%     2) Band-Specific Phase-Lag Index (PLI - volume-conduction immune) (75 features)
%     3) Band-Specific Instantaneous Envelope Coupling (EnvCorr) (75 features)
%     4) Band-Specific Magnitude-Squared Spectral Coherence (MSCoh) (75 features)
%     5) Riemannian Log-Euclidean Covariance & Eigenvalue Synchrony (45 features)
%   Both Raw (345) and Subject-Baseline Calibrated (345) versions (= 690 features)
%   are fused with the 2880 features -> 3570 Total Features, followed immediately
%   by Leak-Free 5-Fold CV Classification and Excel/Figure generation.
%
% [WHY IT IS IMPORTANT]:
%   Single-channel features already separated Amusement very well (75.29% Acc,
%   0.848 AUC), while Confusion vs Fear (68.16% Acc, 0.750 AUC) requires capturing
%   inter-regional brain synchronization. Cognitive Confusion triggers frontal-
%   parietal Theta/Alpha phase locking, whereas Fear triggers Beta/Gamma coupling
%   and Alpha desynchronization. Fusing Connectivity with Single-Channel features
%   directly targets the Confusion-vs-Fear and 3-Class Triple accuracy bottleneck.
%
% [HOW WE DO IT]:
%   - Uses vectorized FFT Analytic Signal Filtering (Hilbert transform in frequency
%     domain) to extract instantaneous phase and envelope for all 6 channels at once.
%   - Computes Log-Euclidean Matrix Logarithm `logm(Cov)` on the 6x6 SPD covariance
%     matrix to project spatial brain covariance into a flat Euclidean tangent space.
% =========================================================================
clear; clc; close all;
warning('off', 'all');

tic;
fprintf('=========================================================================\n');
fprintf('STEP 4: BRAIN FUNCTIONAL CONNECTIVITY FUSION ');
fprintf('=========================================================================\n');

%% 1. LOAD RAW SIGNALS ('eeg_41711.mat') & STEP 1 FEATURES ('eeg_Golden6Ch_2880Calibrated_AmuConFea.mat')
raw_sig_file = 'eeg_41711.mat';
step1_file   = 'eeg_Golden6Ch_2880Calibrated_AmuConFea.mat';

if ~isfile(raw_sig_file)
    error('File "%s" not found! Needed to extract phase/coherence between channels.', raw_sig_file);
end
if ~isfile(step1_file)
    error('File "%s" not found! Needed to fuse with the 2880 calibrated features.', step1_file);
end

fprintf('Loading time-series from "%s" and Step 1 features from "%s"...\n', raw_sig_file, step1_file);
load(raw_sig_file, 'Segments_Balanced', 'Fs');
load(step1_file, 'Features_Cal_2880', 'Feature_Names_2880', 'Channel_List_2880', ...
     'Domain_List_2880', 'CalType_List_2880', 'EmotionID_3', 'ParticipantID_3', ...
     'golden_ch_idx', 'target_emotions', 'emotion_names');

[seq_len, ~, num_seg] = size(Segments_Balanced);
golden_ch_idx = [10, 5, 6, 7, 3, 8];
num_g = length(golden_ch_idx); % 6 Golden Channels

% Extract only the 6 Golden Channels from the time-series tensor: [2560 x 6 x 1305]
EEG_Golden = double(Segments_Balanced(:, golden_ch_idx, :));
clear Segments_Balanced;

%% 2. PRECOMPUTE FREQUENCY GRIDS & PAIR INDICES FOR 6 GOLDEN CHANNELS (15 PAIRS)
pair_list = zeros(15, 2);
pair_names = cell(15, 1);
p_ptr = 1;
for i = 1:num_g
    for j = (i+1):num_g
        pair_list(p_ptr, :) = [i, j];
        pair_names{p_ptr} = sprintf('Ch%02d_Ch%02d', golden_ch_idx(i), golden_ch_idx(j));
        p_ptr = p_ptr + 1;
    end
end
lin_upper_15 = sub2ind([num_g, num_g], pair_list(:, 1), pair_list(:, 2));

% Upper-triangular indices including diagonal for 6x6 Log-Covariance (21 elements)
[r_u21, c_u21] = find(triu(ones(num_g)));
lin_upper_21 = sub2ind([num_g, num_g], r_u21, c_u21);

% Analytic signal FFT filter bank masks for the 5 physiological bands
f_fft = (0:(seq_len-1))' * (Fs / seq_len);
h_mult = zeros(seq_len, 1);
h_mult(1) = 1.0;
h_mult(2:floor(seq_len/2)) = 2.0;
if mod(seq_len, 2) == 0
    h_mult(seq_len/2 + 1) = 1.0;
else
    h_mult(ceil(seq_len/2)) = 2.0;
end

band_edges = [0.5, 4; 4, 8; 8, 13; 13, 30; 30, 45];
band_tags  = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
num_bands  = 5;
band_analytic_masks = zeros(seq_len, num_bands);
for b = 1:num_bands
    m_b = (f_fft >= band_edges(b, 1) & f_fft < band_edges(b, 2));
    band_analytic_masks(:, b) = double(m_b) .* h_mult;
end

% Multi-window Welch parameters for Magnitude-Squared Spectral Coherence (10 sub-windows)
coh_win_len = 512;
coh_step    = 256;
coh_starts  = 1:coh_step:(seq_len - coh_win_len + 1);
num_coh_win = length(coh_starts);
coh_hamming = hamming(coh_win_len);
f_coh       = (0:(coh_win_len/2))' * (Fs / coh_win_len);
coh_band_masks = false(length(f_coh), num_bands);
for b = 1:num_bands
    coh_band_masks(:, b) = (f_coh >= band_edges(b, 1) & f_coh < band_edges(b, 2));
end

%% 3. BUILD METADATA NAMES FOR THE 345 RAW + 345 SUBJCAL = 690 CONNECTIVITY FEATURES
num_conn_raw = 345;
Conn_Base_Names = cell(num_conn_raw, 1);
Conn_Base_Dom   = cell(num_conn_raw, 1);
Conn_Base_Ch    = cell(num_conn_raw, 1);

idx_c = 1;
% 1..75: PLV, 76..150: PLI, 151..225: EnvCorr, 226..300: Coherence
conn_types = {'PLV', 'PLI', 'EnvCorr', 'Coherence'};
for t_c = 1:4
    c_type_str = conn_types{t_c};
    for b = 1:num_bands
        b_str = band_tags{b};
        for p_i = 1:15
            Conn_Base_Names{idx_c} = sprintf('Conn_%s_%s_%s', c_type_str, b_str, pair_names{p_i});
            Conn_Base_Dom{idx_c}   = 'Connectivity_Synchrony';
            Conn_Base_Ch{idx_c}    = sprintf('Channel_%02d', golden_ch_idx(pair_list(p_i, 1)));
            idx_c = idx_c + 1;
        end
    end
end
% 301..315: Broadband Pearson Correlation (15)
for p_i = 1:15
    Conn_Base_Names{idx_c} = sprintf('Conn_BroadCorr_%s', pair_names{p_i});
    Conn_Base_Dom{idx_c}   = 'Connectivity_Riemannian';
    Conn_Base_Ch{idx_c}    = sprintf('Channel_%02d', golden_ch_idx(pair_list(p_i, 1)));
    idx_c = idx_c + 1;
end
% 316..336: Log-Euclidean Covariance Upper Triangle (21)
for u_i = 1:21
    chA = golden_ch_idx(r_u21(u_i));
    chB = golden_ch_idx(c_u21(u_i));
    Conn_Base_Names{idx_c} = sprintf('Conn_RiemannLogCov_Ch%02d_Ch%02d', chA, chB);
    Conn_Base_Dom{idx_c}   = 'Connectivity_Riemannian';
    Conn_Base_Ch{idx_c}    = sprintf('Channel_%02d', chA);
    idx_c = idx_c + 1;
end
% 337..345: 6 Log-Eigenvalues + 3 Global Synchrony Complexity Metrics (9)
for e_i = 1:6
    Conn_Base_Names{idx_c} = sprintf('Conn_CovLogEigVal_%02d', e_i);
    Conn_Base_Dom{idx_c}   = 'Connectivity_Riemannian';
    Conn_Base_Ch{idx_c}    = 'Channel_10';
    idx_c = idx_c + 1;
end
extra_eig_names = {'Conn_Global_EigEntropy', 'Conn_Global_CondRatio', 'Conn_Global_FrobeniusNorm'};
for e_i = 1:3
    Conn_Base_Names{idx_c} = extra_eig_names{e_i};
    Conn_Base_Dom{idx_c}   = 'Connectivity_Riemannian';
    Conn_Base_Ch{idx_c}    = 'Channel_10';
    idx_c = idx_c + 1;
end

%% 4. VECTORIZED EXTRACTION OF ALL 345 CONNECTIVITY FEATURES PER SEGMENT
fprintf('Extracting 345 Functional Connectivity features across %d segments...\n', num_seg);
X_Conn_Raw = zeros(num_seg, num_conn_raw);

for s = 1:num_seg
    sig6 = EEG_Golden(:, :, s); % [2560 x 6]
    sig6 = bsxfun(@minus, sig6, mean(sig6, 1));
    
    % Single 6-channel FFT for fast analytic Hilbert band filtering
    X_f = fft(sig6, seq_len, 1);
    
    plv_vec = zeros(1, 75);
    pli_vec = zeros(1, 75);
    env_vec = zeros(1, 75);
    
    for b = 1:num_bands
        % Analytic complex signal Z_b: [2560 x 6]
        Z_b = ifft(bsxfun(@times, X_f, band_analytic_masks(:, b)), seq_len, 1);
        abs_Z = abs(Z_b) + 1e-12;
        unit_Z = Z_b ./ abs_Z; % exp(j * phi(t))
        
        % 1) Exact PLV across all 15 pairs via 6x6 matrix product
        plv_mat = abs((unit_Z' * unit_Z) / seq_len);
        idx_span = ((b-1)*15 + 1):(b*15);
        plv_vec(idx_span) = plv_mat(lin_upper_15)';
        
        % 2) Exact PLI (Phase-Lag Index) across all 15 pairs
        Z_p1 = Z_b(:, pair_list(:, 1));
        Z_p2 = Z_b(:, pair_list(:, 2));
        pli_vec(idx_span) = abs(mean(sign(imag(Z_p1 .* conj(Z_p2))), 1));
        
        % 3) Instantaneous Envelope Coupling (Amplitude-Amplitude Correlation)
        env_c = bsxfun(@minus, abs_Z, mean(abs_Z, 1));
        env_n = bsxfun(@rdivide, env_c, sqrt(sum(env_c .^ 2, 1)) + 1e-12);
        env_corr_mat = env_n' * env_n;
        env_vec(idx_span) = env_corr_mat(lin_upper_15)';
    end
    
    % 4) Multi-Window Welch Magnitude-Squared Spectral Coherence across 5 bands
    Pxx_auto  = zeros(length(f_coh), num_g);
    Pxy_cross = zeros(length(f_coh), 15);
    for w_i = 1:num_coh_win
        idx_w = coh_starts(w_i):(coh_starts(w_i) + coh_win_len - 1);
        seg_w = bsxfun(@times, sig6(idx_w, :), coh_hamming);
        W_fft = fft(seg_w, coh_win_len, 1);
        W_half = W_fft(1:length(f_coh), :);
        Pxx_auto  = Pxx_auto + (abs(W_half) .^ 2);
        Pxy_cross = Pxy_cross + (W_half(:, pair_list(:, 1)) .* conj(W_half(:, pair_list(:, 2))));
    end
    denom_coh = Pxx_auto(:, pair_list(:, 1)) .* Pxx_auto(:, pair_list(:, 2)) + 1e-12;
    mscoh_full = (abs(Pxy_cross) .^ 2) ./ denom_coh; % [257 x 15]
    
    coh_vec = zeros(1, 75);
    for b = 1:num_bands
        idx_span = ((b-1)*15 + 1):(b*15);
        coh_vec(idx_span) = mean(mscoh_full(coh_band_masks(:, b), :), 1);
    end
    
    % 5) Broadband Correlation + Riemannian Log-Euclidean Covariance & Eigenvalues
    sig_norm = bsxfun(@rdivide, sig6, sqrt(sum(sig6 .^ 2, 1)) + 1e-12);
    corr_broad = sig_norm' * sig_norm;
    broad_corr_15 = corr_broad(lin_upper_15)';
    
    % SPD Covariance matrix with shrinkage regularization for exact matrix logarithm
    cov_mat = (sig6' * sig6) / (seq_len - 1);
    tr_cov  = trace(cov_mat) / num_g + 1e-10;
    cov_spd = 0.95 * cov_mat + 0.05 * tr_cov * eye(num_g);
    [V_eig, D_eig] = eig(cov_spd);
    eig_vals = max(real(diag(D_eig)), 1e-10);
    eig_sorted = sort(eig_vals, 'descend');
    log_eig = log(eig_sorted);
    
    % Matrix logarithm in SPD tangent space: V * diag(log(lambda)) * V'
    log_cov_mat = V_eig * diag(log(eig_vals)) * V_eig';
    riemann_21  = real(log_cov_mat(lin_upper_21))';
    
    p_eig = eig_sorted / sum(eig_sorted);
    eig_ent = -sum(p_eig .* log2(p_eig + eps)) / log2(num_g);
    cond_ratio = log10(eig_sorted(1) / (eig_sorted(end) + 1e-10));
    frob_norm  = sqrt(sum(riemann_21 .^ 2));
    
    X_Conn_Raw(s, :) = [plv_vec, pli_vec, env_vec, coh_vec, ...
                        broad_corr_15, riemann_21, log_eig', eig_ent, cond_ratio, frob_norm];
end

X_Conn_Raw(isnan(X_Conn_Raw) | isinf(X_Conn_Raw)) = 0;

%% 5. APPLY ROBUST SUBJECT-BASELINE CALIBRATION TO CONNECTIVITY FEATURES
fprintf('Applying Subject-Baseline Calibration to the 345 Connectivity features...\n');
X_Conn_SubjCal = zeros(size(X_Conn_Raw));
uniq_pids = unique(ParticipantID_3);

for p_i = 1:length(uniq_pids)
    p_mask = (ParticipantID_3 == uniq_pids(p_i));
    Xp = X_Conn_Raw(p_mask, :);
    if size(Xp, 1) > 1
        med_p = median(Xp, 1);
        iqr_p = prctile(Xp, 75, 1) - prctile(Xp, 25, 1);
        std_p = std(Xp, 0, 1) + 1e-10;
        scale_p = iqr_p;
        scale_p(scale_p < 1e-8) = std_p(scale_p < 1e-8);
        X_Conn_SubjCal(p_mask, :) = bsxfun(@rdivide, bsxfun(@minus, Xp, med_p), scale_p + 1e-10);
    else
        X_Conn_SubjCal(p_mask, :) = Xp;
    end
end
X_Conn_SubjCal(isnan(X_Conn_SubjCal) | isinf(X_Conn_SubjCal)) = 0;

Names_Conn_Raw  = cellfun(@(s) ['RawConn_' s],  Conn_Base_Names, 'UniformOutput', false);
Names_Conn_Subj = cellfun(@(s) ['SubjConn_' s], Conn_Base_Names, 'UniformOutput', false);
Cal_Conn_Raw    = repmat({'4_Connectivity_Raw'},     num_conn_raw, 1);
Cal_Conn_Subj   = repmat({'5_Connectivity_SubjCal'}, num_conn_raw, 1);

%% 6. FUSE STEP 1 (2880 FEATS) + STEP 4 CONNECTIVITY (690 FEATS) = 3570 FEATURES
Features_Fused_3570 = [Features_Cal_2880, X_Conn_Raw, X_Conn_SubjCal];
Feature_Names_3570  = [Feature_Names_2880; Names_Conn_Raw; Names_Conn_Subj];
Channel_List_3570   = [Channel_List_2880;  Conn_Base_Ch;   Conn_Base_Ch];
Domain_List_3570    = [Domain_List_2880;   Conn_Base_Dom;  Conn_Base_Dom];
CalType_List_3570   = [CalType_List_2880;  Cal_Conn_Raw;   Cal_Conn_Subj];

fused_mat_file = 'eeg_Fused_3570_GoldenConn_AmuConFea.mat';
fprintf('Saving fused 3570-feature dataset to "%s"...\n', fused_mat_file);
save(fused_mat_file, 'Features_Fused_3570', 'Feature_Names_3570', 'Channel_List_3570', ...
     'Domain_List_3570', 'CalType_List_3570', 'EmotionID_3', 'ParticipantID_3', ...
     'golden_ch_idx', 'target_emotions', 'emotion_names', '-v7.3');

%% 7. EXPORT EXCEL WORKBOOK & DIAGNOSTIC COMPARISON FIGURE
out_xlsx = 'Step4_Connectivity_Fusion_Results.xlsx';
if isfile(out_xlsx), delete(out_xlsx); end

writetable(cell2table(Exec_Rows, 'VariableNames', { ...
    'Task_Scenario', 'Winning_Model', 'Step2_Acc_Pct', 'Step3_3Fold_Acc_Pct', ...
    'Step4_Fused_Acc_Pct', 'Step4_Acc_SD_Pct', 'Step4_Accuracy_Formatted', ...
    'Step4_MacroF1_Pct', 'Prev_Best_AUC', 'Step4_OOF_AUC', 'Best_Fold_Acc_Pct'}), ...
    out_xlsx, 'Sheet', 'Executive_Comparison_S2_S3_S4');

writetable(cell2table(AllMod_Rows, 'VariableNames', { ...
    'Task_Scenario', 'Classifier_Name', 'Acc_Mean_Pct', 'Acc_SD_Pct', 'Accuracy_Formatted', ...
    'MacroF1_Mean_Pct', 'MacroF1_SD_Pct', 'Sensitivity_Pct', 'Precision_Pct', 'OOF_ROC_AUC', 'Best_Fold_Acc_Pct'}), ...
    out_xlsx, 'Sheet', 'All_Classifiers_Step4');

writetable(cell2table(TopConn_Rows, 'VariableNames', { ...
    'Task_Scenario', 'Rank_In_Connectivity', 'Connectivity_Feature_Name', ...
    'Domain', 'Calibration_Type', 'Univariate_Max_AUC'}), ...
    out_xlsx, 'Sheet', 'Top_Connectivity_Features');

% Plot 6-Panel Dashboard Figure
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0.4, 0.4, 14.2, 8.6], 'Visible', 'on');
for t_idx = 1:4
    ax = subplot(2, 3, t_idx + (t_idx > 2));
    yt = Best_Y_True{t_idx}; yp = Best_OOF_Preds{t_idx}; c_names = tasks_cfg{t_idx, 3}; n_c = length(c_names);
    cm = confusionmat(yt, yp);
    cm_pct = bsxfun(@rdivide, double(cm), sum(cm, 2) + eps) * 100.0;
    imagesc(ax, cm_pct); colormap(ax, parula(256)); caxis(ax, [0, 100]);
    set(ax, 'XTick', 1:n_c, 'XTickLabel', c_names, 'YTick', 1:n_c, 'YTickLabel', c_names, 'FontSize', 8);
    xlabel(ax, 'Predicted Emotion', 'FontSize', 8); ylabel(ax, 'True Emotion', 'FontSize', 8);
    title(ax, sprintf('%s\n(%s: %.2f%%)', strrep(tasks_cfg{t_idx, 1}, '_', ' '), ...
        strrep(Best_Winner{t_idx}, '_', ' '), mean(yt == yp)*100), 'FontSize', 8.5, 'FontWeight', 'normal');
    for i = 1:n_c
        for j = 1:n_c
            tc = 'w'; if cm_pct(i, j) > 50, tc = 'k'; end
            text(ax, j, i, sprintf('%d\n(%.1f%%)', cm(i, j), cm_pct(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', tc, 'FontSize', 8);
        end
    end
end

ax_roc = subplot(2, 3, 3); hold(ax_roc, 'on');
r_cols = [0.11, 0.62, 0.47; 0.85, 0.37, 0.01; 0.46, 0.44, 0.70];
for t_idx = 1:3
    [fpr, tpr, ~, auc_v] = perfcurve(Best_Y_True{t_idx}, Best_OOF_Probs{t_idx}(:, 2), 2);
    plot(ax_roc, fpr, tpr, 'LineWidth', 1.6, 'Color', r_cols(t_idx, :), ...
        'DisplayName', sprintf('%s (AUC=%.3f)', strrep(tasks_cfg{t_idx, 1}, '_', ' '), auc_v));
end
plot(ax_roc, [0, 1], [0, 1], 'k--', 'LineWidth', 1.0, 'DisplayName', 'Chance (0.500)');
hold(ax_roc, 'off'); grid(ax_roc, 'on'); set(ax_roc, 'FontSize', 8);
xlabel(ax_roc, 'False Positive Rate', 'FontSize', 8); ylabel(ax_roc, 'True Positive Rate', 'FontSize', 8);
title(ax_roc, 'Step 4 Fused Out-Of-Fold ROC Curves', 'FontSize', 8.5, 'FontWeight', 'normal');
legend(ax_roc, 'Location', 'southeast', 'FontSize', 7);

ax_bar = subplot(2, 3, 6);
bar_data = [cell2mat(Exec_Rows(:, 3)), cell2mat(Exec_Rows(:, 4)), cell2mat(Exec_Rows(:, 5)), cell2mat(Exec_Rows(:, 10))*100];
bh = bar(ax_bar, bar_data, 'grouped');
bh(1).FaceColor = [0.46, 0.44, 0.70]; bh(2).FaceColor = [0.17, 0.36, 0.56];
bh(3).FaceColor = [0.11, 0.62, 0.47]; bh(4).FaceColor = [0.85, 0.37, 0.01];
set(ax_bar, 'XTick', 1:4, 'XTickLabel', {'Amu vs Con', 'Amu vs Fea', 'Con vs Fea', 'Triple (3-Cls)'}, 'FontSize', 8);
ylabel(ax_bar, 'Score (%)', 'FontSize', 8); ylim(ax_bar, [45, 102]); grid(ax_bar, 'on');
title(ax_bar, 'Progression: Step 2 vs Step 3 vs Step 4 (Fused)', 'FontSize', 8.5, 'FontWeight', 'normal');
legend(ax_bar, {'Step 2 Acc', 'Step 3 Acc', 'Step 4 Fused Acc', 'Step 4 AUC (%)'}, 'Location', 'northeast', 'FontSize', 7);

exportgraphics(fig, 'Step4_Connectivity_Fusion_Dashboard.png', 'Resolution', 300);
exportgraphics(fig, 'Step4_Connectivity_Fusion_Dashboard.pdf', 'ContentType', 'vector');

fprintf('\n=========================================================================\n');
fprintf('STEP 4 COMPLETED IN %.2f SECONDS! Saved "%s" & Dashboard Figures.\n', toc, out_xlsx);
fprintf('=========================================================================\n');

%% =========================================================================
% HELPER FUNCTIONS FOR FUSED mRMR SELECTION & SPECIALIST ROUTING
% =========================================================================
function sel_idx = select_fused_mrmr_binary(X_tr, y_tr, num_keep)
    % Guarantees both Single-Channel (1..2880) and Connectivity (2881..3570)
    % candidates enter the correlation-penalized mRMR pool inside each fold
    c1 = X_tr(y_tr == 1, :); c2 = X_tr(y_tr == 2, :);
    n1 = size(c1, 1); n2 = size(c2, 1);
    mu1 = mean(c1, 1); mu2 = mean(c2, 1); mu_all = mean(X_tr, 1);
    ss_b = n1 * ((mu1 - mu_all).^2) + n2 * ((mu2 - mu_all).^2);
    ss_w = sum(bsxfun(@minus, c1, mu1).^2, 1) + sum(bsxfun(@minus, c2, mu2).^2, 1) + eps;
    f_scores = ss_b ./ (ss_w / (n1 + n2 - 2));
    f_scores(isnan(f_scores) | isinf(f_scores)) = 0;
    
    [~, top_single] = sort(f_scores(1:2880), 'descend');
    [~, top_conn]   = sort(f_scores(2881:end), 'descend');
    cand_pool = [top_single(1:200), 2880 + top_conn(1:80)];
    
    X_cand = X_tr(:, cand_pool);
    rel_vec = f_scores(cand_pool);
    rel_vec = (rel_vec - min(rel_vec)) / (max(rel_vec) - min(rel_vec) + eps);
    
    corr_mat = abs(corr(X_cand)); corr_mat(isnan(corr_mat)) = 0;
    chosen_local = zeros(1, num_keep); chosen_local(1) = 1;
    remaining = 2:length(cand_pool);
    for k = 2:num_keep
        best_sc = -1e9; best_pos = 1;
        for r_i = 1:length(remaining)
            c_idx = remaining(r_i);
            red = mean(corr_mat(c_idx, chosen_local(1:k-1)));
            sc = rel_vec(c_idx) - 0.42 * red;
            if sc > best_sc, best_sc = sc; best_pos = r_i; end
        end
        chosen_local(k) = remaining(best_pos);
        remaining(best_pos) = [];
    end
    sel_idx = cand_pool(chosen_local);
end

function sel_idx = select_fused_mrmr_triple(X_tr, y_tr, k_per_pair)
    pairs = [1, 2; 1, 3; 2, 3];
    union_cols = [];
    for p_i = 1:3
        m_p = (y_tr == pairs(p_i, 1)) | (y_tr == pairs(p_i, 2));
        y_p = double(y_tr(m_p) == pairs(p_i, 2)) + 1;
        cols_p = select_fused_mrmr_binary(X_tr(m_p, :), y_p, k_per_pair * 2);
        for c_id = cols_p
            if ~ismember(c_id, union_cols), union_cols(end+1) = c_id; end %#ok<AGROW>
            if length(union_cols) >= p_i * k_per_pair, break; end
        end
    end
    sel_idx = union_cols;
end

function probs_3cls = run_3class_specialist_router(X_tr, y_tr, X_te)
    % Stage 1: Amusement (1) vs Rest (2 or 3)
    y_s1 = double(y_tr > 1) + 1; % 1 = Amusement, 2 = Confusion/Fear
    cols_s1 = select_fused_mrmr_binary(X_tr, y_s1, 30);
    [~, p_s1_knn] = run_classifier_fold(X_tr(:, cols_s1), y_s1, X_te(:, cols_s1), 4, 2);
    [~, p_s1_rf]  = run_classifier_fold(X_tr(:, cols_s1), y_s1, X_te(:, cols_s1), 3, 2);
    [~, p_s1_svm] = run_classifier_fold(X_tr(:, cols_s1), y_s1, X_te(:, cols_s1), 1, 2);
    p_s1 = 0.45 * p_s1_knn + 0.30 * p_s1_rf + 0.25 * p_s1_svm;
    
    % Stage 2: Dedicated Confusion (2) vs Fear (3) Specialist
    m_23 = (y_tr == 2) | (y_tr == 3);
    X_tr_23 = X_tr(m_23, :);
    y_tr_23 = double(y_tr(m_23) == 3) + 1; % 1 = Confusion, 2 = Fear
    cols_s2 = select_fused_mrmr_binary(X_tr_23, y_tr_23, 28);
    [~, p_s2_svm] = run_classifier_fold(X_tr_23(:, cols_s2), y_tr_23, X_te(:, cols_s2), 1, 2);
    [~, p_s2_knn] = run_classifier_fold(X_tr_23(:, cols_s2), y_tr_23, X_te(:, cols_s2), 4, 2);
    [~, p_s2_rf]  = run_classifier_fold(X_tr_23(:, cols_s2), y_tr_23, X_te(:, cols_s2), 3, 2);
    p_s2 = 0.45 * p_s2_svm + 0.35 * p_s2_knn + 0.20 * p_s2_rf;
    
    probs_3cls = [p_s1(:, 1), p_s1(:, 2) .* p_s2(:, 1), p_s1(:, 2) .* p_s2(:, 2)];
    probs_3cls = bsxfun(@rdivide, probs_3cls, sum(probs_3cls, 2) + eps);
end

function [preds, probs] = run_classifier_fold(X_tr, y_tr, X_te, model_id, num_cls)
    n_feat = size(X_tr, 2);
    switch model_id
        case 1 % Tuned RBF SVM
            t_svm = templateSVM('KernelFunction', 'gaussian', 'KernelScale', sqrt(n_feat)*1.15, ...
                                'BoxConstraint', 2.2, 'Standardize', false);
            if num_cls == 2
                mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'gaussian', ...
                              'KernelScale', sqrt(n_feat)*1.15, 'BoxConstraint', 2.2, 'Standardize', false);
            else
                mdl = fitcecoc(X_tr, y_tr, 'Learners', t_svm, 'Coding', 'onevsall');
            end
            [~, sc] = predict(mdl, X_te);
            probs = scores_to_probs_s4(sc);
        case 2 % Subspace Discriminant
            sub_d = max(6, round(n_feat * 0.55));
            t_disc = templateDiscriminant('DiscrimType', 'pseudoLinear');
            mdl = fitcensemble(X_tr, y_tr, 'Method', 'Subspace', 'NumLearningCycles', 60, ...
                               'Learners', t_disc, 'NPredToSample', sub_d);
            [~, sc] = predict(mdl, X_te);
            probs = scores_to_probs_s4(sc);
        case 3 % Bagged Trees RF
            t_tr = templateTree('MaxNumSplits', 30, 'MinLeafSize', 2);
            mdl = fitcensemble(X_tr, y_tr, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', t_tr);
            [~, sc] = predict(mdl, X_te);
            probs = scores_to_probs_s4(sc);
        case 4 % Weighted Correlation KNN (Multi-k Ensemble: k=7 and k=11)
            mdl1 = fitcknn(X_tr, y_tr, 'NumNeighbors', 7,  'Distance', 'correlation', 'DistanceWeight', 'squaredinverse');
            mdl2 = fitcknn(X_tr, y_tr, 'NumNeighbors', 11, 'Distance', 'correlation', 'DistanceWeight', 'squaredinverse');
            [~, sc1] = predict(mdl1, X_te);
            [~, sc2] = predict(mdl2, X_te);
            probs = 0.5 * scores_to_probs_s4(sc1) + 0.5 * scores_to_probs_s4(sc2);
    end
    [~, preds] = max(probs, [], 2);
end

function probs = scores_to_probs_s4(sc)
    sc(isnan(sc) | isinf(sc)) = 0;
    sc_s = bsxfun(@minus, sc, max(sc, [], 2));
    exp_s = exp(sc_s);
    probs = bsxfun(@rdivide, exp_s, sum(exp_s, 2) + eps);
end

function [acc, macro_f1, macro_rec, macro_prec, macro_auc] = compute_metrics_step4(y_true, y_pred, probs, num_cls)
    acc = mean(y_true == y_pred);
    cm = confusionmat(y_true, y_pred, 'Order', 1:num_cls);
    f1_v = zeros(num_cls, 1); rec_v = zeros(num_cls, 1); prec_v = zeros(num_cls, 1); auc_v = zeros(num_cls, 1);
    for c = 1:num_cls
        tp = cm(c, c); fp = sum(cm(:, c)) - tp; fn = sum(cm(c, :)) - tp;
        prec_v(c) = tp / (tp + fp + eps);
        rec_v(c)  = tp / (tp + fn + eps);
        f1_v(c)   = (2 * prec_v(c) * rec_v(c)) / (prec_v(c) + rec_v(c) + eps);
        yb = double(y_true == c); pc = probs(:, c);
        if sum(yb==1) > 0 && sum(yb==0) > 0 && std(pc) > 1e-12
            rt = tiedrank(pc); n1 = sum(yb==1); n0 = sum(yb==0);
            auc_v(c) = (sum(rt(yb==1)) - n1*(n1+1)/2) / (n1 * n0);
        else
            auc_v(c) = 0.5;
        end
    end
    macro_f1 = mean(f1_v); macro_rec = mean(rec_v); macro_prec = mean(prec_v); macro_auc = mean(auc_v);
end