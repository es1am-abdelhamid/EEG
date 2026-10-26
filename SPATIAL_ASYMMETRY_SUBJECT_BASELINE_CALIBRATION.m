%% =========================================================================
% STEP 1: GOLDEN-CHANNEL SPATIAL, ASYMMETRY & SUBJECT-BASELINE CALIBRATION
% =========================================================================
% [WHAT WE ARE DOING ]:
%   We load the pre-extracted 2240 features ('eeg_2240features_AmuConFea.mat'),
%   focus on the 6 Golden Channels [Ch10, Ch05, Ch06, Ch07, Ch03, Ch08] that
%   captured 100% of the Top-100 features, and generate 3 calibrated feature
%   representations (Total = 2880 calibrated features):
%     1) Within-Segment Spatial Relative Features (960 features)
%     2) Cross-Channel Differential Asymmetry Features (960 features)
%     3) Robust Subject-Baseline Calibrated Features (960 features)
%
% [WHY IT IS IMPORTANT ]:
%   Raw EEG features suffer from massive Inter-Subject Baseline Drift (due to
%   differences in skull thickness, scalp impedance, and baseline resting EEG),
%   which inflates standard deviation and caps raw univariate AUC at ~0.627.
%   Calibrating each channel relative to the rest of the brain within the same
%   segment and across symmetric channel pairs removes anatomical noise and
%   isolates the pure neurophysiological signature of each emotion.
%
% [HOW WE DO IT ]:
%   - Reshaping the 2240 matrix into a 3D tensor: [N_Segments x 14_Channels x 160_Feats].
%   - Computing 100% leak-free Within-Segment Spatial Z-Scores across the 14 channels.
%   - Computing pairwise spatial differences across the Top-4 Core Channels
%     (Ch10, Ch05, Ch06, Ch07 -> 6 pairs x 160 features = 960 Asymmetry features).
%   - Computing Robust Median/IQR standardization per participant across their
%     strictly balanced segments (960 Subject-Calibrated features).
%   - Re-evaluating all 2880 calibrated features with the 5 statistical tests
%     (AUC, Pearson, ANOVA, Mann-Whitney/Kruskal, KS-Test) and plotting the
%     updated 4-panel diagnostic figure comparing against 0.574 and 0.627.
% =========================================================================
clear; clc; close all;
warning('off', 'all');

tic;
fprintf('=========================================================================\n');
fprintf('STEP 1: SPATIAL, ASYMMETRY & SUBJECT CALIBRATION ON THE 6 GOLDEN CHANNELS\n');
fprintf('=========================================================================\n');

% 1. LOAD THE PRE-EXTRACTED 2240-FEATURE DATASET
in_file = 'eeg_2240features_AmuConFea.mat';
if ~isfile(in_file)
    error('File "%s" not found! Please run the 2240-feature script first.', in_file);
end

fprintf('Loading "%s"...\n', in_file);
load(in_file); % Loads: Features_2240, Feature_Names_2240, EmotionID_3, ParticipantID_3, etc.

[num_seg, total_raw_feats] = size(Features_2240);
num_ch = 14;
num_feat_per_ch = 160; % 40 Time + 40 Freq + 40 DB4 + 40 COIF5

% Extract base 160 feature names and domains from Channel_01
base_160_names   = BaseName_Per_Feature(1:num_feat_per_ch);
base_160_domains = Domain_Per_Feature(1:num_feat_per_ch);

% Reshape 2D matrix [N_seg x 2240] into 3D tensor: [N_seg x 14_Ch x 160_Feats]
X_3D_Raw = zeros(num_seg, num_ch, num_feat_per_ch);
for ch = 1:num_ch
    c_start = (ch - 1) * num_feat_per_ch + 1;
    c_end   = ch * num_feat_per_ch;
    X_3D_Raw(:, ch, :) = Features_2240(:, c_start:c_end);
end

% Define the 6 Golden Channels (100% of Top-100) and Top-4 Core Channels (94% of Top-100)
golden_ch_idx  = [10, 5, 6, 7, 3, 8];
core4_ch_idx   = [10, 5, 6, 7];
num_golden     = length(golden_ch_idx); % 6 channels

%% 2. TRANSFORMATION A: WITHIN-SEGMENT SPATIAL RELATIVE Z-SCORE (100% LEAK-FREE)
% For each segment s and feature f, normalize across the 14 channels so that
% global segment-wide amplitude scaling and impedance differences cancel out.
fprintf('1/3 Computing Within-Segment Spatial Relative Features (960 features)...\n');

seg_ch_mean = mean(X_3D_Raw, 2);               % [N_seg x 1 x 160]
seg_ch_std  = std(X_3D_Raw, 0, 2) + 1e-10;     % [N_seg x 1 x 160]
X_3D_SpatialAll = bsxfun(@rdivide, bsxfun(@minus, X_3D_Raw, seg_ch_mean), seg_ch_std);

% Keep the 6 Golden Channels: [N_seg x 6 x 160] -> reshape to [N_seg x 960]
X_Spatial_Golden = zeros(num_seg, num_golden * num_feat_per_ch);
Names_Spatial    = cell(num_golden * num_feat_per_ch, 1);
Ch_Spatial       = cell(num_golden * num_feat_per_ch, 1);
Dom_Spatial      = cell(num_golden * num_feat_per_ch, 1);
Type_Spatial     = cell(num_golden * num_feat_per_ch, 1);

ptr = 1;
for g_i = 1:num_golden
    ch = golden_ch_idx(g_i);
    ch_str = sprintf('Channel_%02d', ch);
    for f_i = 1:num_feat_per_ch
        X_Spatial_Golden(:, ptr) = X_3D_SpatialAll(:, ch, f_i);
        Names_Spatial{ptr} = sprintf('SpatialRel_Ch%02d_%s', ch, base_160_names{f_i});
        Ch_Spatial{ptr}    = ch_str;
        Dom_Spatial{ptr}   = base_160_domains{f_i};
        Type_Spatial{ptr}  = '1_Spatial_Relative';
        ptr = ptr + 1;
    end
end

%% 3. TRANSFORMATION B: CROSS-CHANNEL DIFFERENTIAL ASYMMETRY (100% LEAK-FREE)
% Compute pairwise spatial differences among the Top-4 Core Channels [10, 5, 6, 7]
% (6 pairs x 160 features = 960 features) within each single segment.
fprintf('2/3 Computing Cross-Channel Asymmetry Features across Core Channels (960 features)...\n');

core_pairs = [10, 5; 10, 6; 10, 7; 5, 6; 5, 7; 6, 7];
num_pairs  = size(core_pairs, 1); % 6 pairs

X_Asym_Golden = zeros(num_seg, num_pairs * num_feat_per_ch);
Names_Asym    = cell(num_pairs * num_feat_per_ch, 1);
Ch_Asym       = cell(num_pairs * num_feat_per_ch, 1);
Dom_Asym      = cell(num_pairs * num_feat_per_ch, 1);
Type_Asym     = cell(num_pairs * num_feat_per_ch, 1);

ptr = 1;
for p_i = 1:num_pairs
    cA = core_pairs(p_i, 1);
    cB = core_pairs(p_i, 2);
    pair_ch_str = sprintf('Ch%02d_vs_Ch%02d', cA, cB);
    for f_i = 1:num_feat_per_ch
        % Differential contrast between spatially normalized channels A and B
        X_Asym_Golden(:, ptr) = X_3D_SpatialAll(:, cA, f_i) - X_3D_SpatialAll(:, cB, f_i);
        Names_Asym{ptr} = sprintf('Asym_%s_%s', pair_ch_str, base_160_names{f_i});
        Ch_Asym{ptr}    = sprintf('Channel_%02d', cA); % Primary anchor channel
        Dom_Asym{ptr}   = base_160_domains{f_i};
        Type_Asym{ptr}  = '2_CrossChannel_Asymmetry';
        ptr = ptr + 1;
    end
end

%% 4. TRANSFORMATION C: ROBUST SUBJECT-BASELINE CALIBRATION (960 FEATURES)
% Remove inter-participant skull/impedance offset using robust Median & IQR
% across each participant's equally balanced segments on the 6 Golden Channels.
fprintf('3/3 Computing Robust Subject-Baseline Calibrated Features (960 features)...\n');

X_Raw_Golden = zeros(num_seg, num_golden * num_feat_per_ch);
Names_Subj   = cell(num_golden * num_feat_per_ch, 1);
Ch_Subj      = cell(num_golden * num_feat_per_ch, 1);
Dom_Subj     = cell(num_golden * num_feat_per_ch, 1);
Type_Subj    = cell(num_golden * num_feat_per_ch, 1);

ptr = 1;
for g_i = 1:num_golden
    ch = golden_ch_idx(g_i);
    ch_str = sprintf('Channel_%02d', ch);
    for f_i = 1:num_feat_per_ch
        X_Raw_Golden(:, ptr) = X_3D_Raw(:, ch, f_i);
        Names_Subj{ptr} = sprintf('SubjCal_Ch%02d_%s', ch, base_160_names{f_i});
        Ch_Subj{ptr}    = ch_str;
        Dom_Subj{ptr}   = base_160_domains{f_i};
        Type_Subj{ptr}  = '3_Subject_Calibrated';
        ptr = ptr + 1;
    end
end

X_Subj_Golden = zeros(size(X_Raw_Golden));
uniq_pids = unique(ParticipantID_3);
for p_i = 1:length(uniq_pids)
    p_mask = (ParticipantID_3 == uniq_pids(p_i));
    X_p = X_Raw_Golden(p_mask, :);
    if size(X_p, 1) > 1
        med_p = median(X_p, 1);
        q75_p = prctile(X_p, 75, 1);
        q25_p = prctile(X_p, 25, 1);
        iqr_p = (q75_p - q25_p);
        std_p = std(X_p, 0, 1) + 1e-10;
        % Use IQR where non-zero, fallback to std for discrete/sparse features
        scale_p = iqr_p;
        zero_iqr = (scale_p < 1e-8);
        scale_p(zero_iqr) = std_p(zero_iqr);
        X_Subj_Golden(p_mask, :) = bsxfun(@rdivide, bsxfun(@minus, X_p, med_p), scale_p + 1e-10);
    else
        X_Subj_Golden(p_mask, :) = X_p;
    end
end

%% 5. COMBINE ALL 2880 CALIBRATED GOLDEN-CHANNEL FEATURES & SAVE .MAT
Features_Cal_2880 = [X_Spatial_Golden, X_Asym_Golden, X_Subj_Golden];
Features_Cal_2880(isnan(Features_Cal_2880) | isinf(Features_Cal_2880)) = 0;

Feature_Names_2880 = [Names_Spatial; Names_Asym; Names_Subj];
Channel_List_2880  = [Ch_Spatial;    Ch_Asym;    Ch_Subj];
Domain_List_2880   = [Dom_Spatial;   Dom_Asym;   Dom_Subj];
CalType_List_2880  = [Type_Spatial;  Type_Asym;  Type_Subj];

total_cal_features = size(Features_Cal_2880, 2); % 2880

cal_mat_file = 'eeg_Golden6Ch_2880Calibrated_AmuConFea.mat';
fprintf('\nSaving calibrated dataset (%d segments x %d features) to "%s"...\n', ...
    num_seg, total_cal_features, cal_mat_file);
save(cal_mat_file, 'Features_Cal_2880', 'Feature_Names_2880', 'Channel_List_2880', ...
     'Domain_List_2880', 'CalType_List_2880', 'EmotionID_3', 'ParticipantID_3', ...
     'golden_ch_idx', 'target_emotions', 'emotion_names', '-v7.3');

%% 6. RUN MULTI-TEST STATISTICAL EVALUATION ON THE 2880 CALIBRATED FEATURES
fprintf('\nRunning 5 Statistical Tests on all %d Calibrated Features...\n', total_cal_features);

pairs_cfg = {
    4,  11, 'Amusement', 'Confusion', 'Amu_vs_Con';
    4,  17, 'Amusement', 'Fear',      'Amu_vs_Fea';
    11, 17, 'Confusion', 'Fear',      'Con_vs_Fea'
};

TOP_N = 100;
file_master_cal = 'Master_Summary_Top100_Calibrated_Step1.xlsx';
if isfile(file_master_cal), delete(file_master_cal); end

AllTasks_MaxAUC  = zeros(total_cal_features, 4);
AllTasks_AbsCorr = zeros(total_cal_features, 4);
AllTasks_AnovaF  = zeros(total_cal_features, 4);
AllTasks_MW_KW_Z = zeros(total_cal_features, 4);
AllTasks_KS_Stat = zeros(total_cal_features, 4);

for p_idx = 1:size(pairs_cfg, 1)
    id1   = pairs_cfg{p_idx, 1};
    id2   = pairs_cfg{p_idx, 2};
    name1 = pairs_cfg{p_idx, 3};
    name2 = pairs_cfg{p_idx, 4};
    s_tag = pairs_cfg{p_idx, 5};
    
    mask1 = (EmotionID_3 == id1); mask2 = (EmotionID_3 == id2);
    mask_pair = mask1 | mask2;
    
    X_pair = Features_Cal_2880(mask_pair, :);
    y_pair = double(EmotionID_3(mask_pair) == id2);
    
    X1 = Features_Cal_2880(mask1, :); X2 = Features_Cal_2880(mask2, :);
    n1 = size(X1, 1); n2 = size(X2, 1);
    
    Mean1 = mean(X1, 1)'; SD1 = std(X1, 0, 1)';
    Mean2 = mean(X2, 1)'; SD2 = std(X2, 0, 1)';
    
    Raw_AUC = zeros(total_cal_features, 1); Max_AUC = zeros(total_cal_features, 1);
    MW_PVal = ones(total_cal_features, 1);  MW_ZScore = zeros(total_cal_features, 1);
    Pearson_r = zeros(total_cal_features, 1); Abs_Pearson = zeros(total_cal_features, 1); Pearson_PVal = ones(total_cal_features, 1);
    ANOVA_F = zeros(total_cal_features, 1); ANOVA_PVal = ones(total_cal_features, 1); Eta_Sq = zeros(total_cal_features, 1);
    KS_Stat = zeros(total_cal_features, 1); KS_PVal = ones(total_cal_features, 1);
    
    for f_i = 1:total_cal_features
        col_all = X_pair(:, f_i); c1 = X1(:, f_i); c2 = X2(:, f_i);
        if std(col_all) < 1e-12, Raw_AUC(f_i)=0.5; Max_AUC(f_i)=0.5; continue; end
        
        r_tied = tiedrank(col_all);
        R2 = sum(r_tied(y_pair == 1));
        auc_val = (R2 - (n2 * (n2 + 1)) / 2.0) / (n1 * n2);
        Raw_AUC(f_i) = auc_val; Max_AUC(f_i) = max(auc_val, 1.0 - auc_val);
        
        [p_mw, ~, stats_mw] = ranksum(c1, c2);
        MW_PVal(f_i) = p_mw;
        if isfield(stats_mw, 'zval'), MW_ZScore(f_i) = abs(stats_mw.zval);
        else, MW_ZScore(f_i) = abs(norminv(max(min(p_mw / 2, 0.499999), 1e-300))); end
        
        [r_mat, p_mat] = corrcoef(col_all, y_pair);
        Pearson_r(f_i) = r_mat(1, 2); Abs_Pearson(f_i) = abs(r_mat(1, 2)); Pearson_PVal(f_i) = p_mat(1, 2);
        
        grand_mean = mean(col_all);
        ss_between = n1 * ((Mean1(f_i) - grand_mean)^2) + n2 * ((Mean2(f_i) - grand_mean)^2);
        ss_within  = sum((c1 - Mean1(f_i)).^2) + sum((c2 - Mean2(f_i)).^2) + eps;
        f_stat     = (ss_between / 1) / (ss_within / (n1 + n2 - 2));
        ANOVA_F(f_i) = f_stat; ANOVA_PVal(f_i) = 1.0 - fcdf(f_stat, 1, n1 + n2 - 2);
        Eta_Sq(f_i)  = ss_between / (ss_between + ss_within + eps);
        
        [~, p_ks, ks_st] = kstest2(c1, c2);
        KS_Stat(f_i) = ks_st; KS_PVal(f_i) = p_ks;
    end
    
    AllTasks_MaxAUC(:, p_idx)  = Max_AUC;
    AllTasks_AbsCorr(:, p_idx) = Abs_Pearson;
    AllTasks_AnovaF(:, p_idx)  = ANOVA_F;
    AllTasks_MW_KW_Z(:, p_idx) = MW_ZScore;
    AllTasks_KS_Stat(:, p_idx) = KS_Stat;
    
    Consensus_Pair = 0.25*norm_01(Max_AUC) + 0.20*norm_01(Abs_Pearson) + ...
                     0.20*norm_01(ANOVA_F) + 0.20*norm_01(MW_ZScore) + 0.15*norm_01(KS_Stat);
                 
    T_Pair = table(Feature_Names_2880, Channel_List_2880, Domain_List_2880, CalType_List_2880, ...
        Mean1, SD1, Mean2, SD2, Raw_AUC, Max_AUC, Pearson_r, Abs_Pearson, Pearson_PVal, ...
        ANOVA_F, Eta_Sq, ANOVA_PVal, MW_ZScore, MW_PVal, KS_Stat, KS_PVal, Consensus_Pair, ...
        'VariableNames', {'Full_Feature_Name', 'Channel', 'Domain', 'Calibration_Type', ...
        [name1 '_Mean'], [name1 '_SD'], [name2 '_Mean'], [name2 '_SD'], ...
        'Raw_AUC', 'Discriminative_Max_AUC', 'Pearson_r', 'Abs_Pearson_r', 'Pearson_P_Value', ...
        'ANOVA_F_Score', 'Eta_Squared_Effect', 'ANOVA_P_Value', ...
        'MannWhitney_Abs_Z', 'MannWhitney_P_Value', 'KS_Statistic', 'KS_P_Value', 'Consensus_Score'});
    
    write_top_n_sheet(T_Pair, 'Consensus_Score', TOP_N, file_master_cal, ['Top100_' s_tag]);
    fprintf('  -> Pair %s: Peak Calibrated AUC = %.4f | Peak ANOVA F = %.2f (p = %.2e)\n', ...
        s_tag, max(Max_AUC), max(ANOVA_F), min(ANOVA_PVal));
end

% Triple (3-Class) Evaluation
mask_c1 = (EmotionID_3 == 4); mask_c2 = (EmotionID_3 == 11); mask_c3 = (EmotionID_3 == 17);
X_c1 = Features_Cal_2880(mask_c1, :); n_c1 = size(X_c1, 1);
X_c2 = Features_Cal_2880(mask_c2, :); n_c2 = size(X_c2, 1);
X_c3 = Features_Cal_2880(mask_c3, :); n_c3 = size(X_c3, 1);

Mean_Amu = mean(X_c1, 1)'; Mean_Con = mean(X_c2, 1)'; Mean_Fea = mean(X_c3, 1)';
Triple_Mean_OvR_AUC = mean(AllTasks_MaxAUC(:, 1:3), 2);
Triple_Max_Pair_AUC = max(AllTasks_MaxAUC(:, 1:3), [], 2);
Triple_Mean_AbsCorr = mean(AllTasks_AbsCorr(:, 1:3), 2);
Triple_Max_KS       = max(AllTasks_KS_Stat(:, 1:3), [], 2);

Triple_ANOVA_F = zeros(total_cal_features, 1); Triple_ANOVA_PVal = ones(total_cal_features, 1);
Triple_KW_Chi2 = zeros(total_cal_features, 1); Triple_KW_PVal = ones(total_cal_features, 1);

N_all = num_seg;
for f_i = 1:total_cal_features
    col_f = Features_Cal_2880(:, f_i);
    if std(col_f) < 1e-12, continue; end
    g_mean = mean(col_f);
    ss_b = n_c1*((Mean_Amu(f_i)-g_mean)^2) + n_c2*((Mean_Con(f_i)-g_mean)^2) + n_c3*((Mean_Fea(f_i)-g_mean)^2);
    ss_w = sum((X_c1(:, f_i)-Mean_Amu(f_i)).^2) + sum((X_c2(:, f_i)-Mean_Con(f_i)).^2) + sum((X_c3(:, f_i)-Mean_Fea(f_i)).^2) + eps;
    f_val = (ss_b / 2.0) / (ss_w / (N_all - 3));
    Triple_ANOVA_F(f_i) = f_val; Triple_ANOVA_PVal(f_i) = 1.0 - fcdf(f_val, 2, N_all - 3);
    
    r_all = tiedrank(col_f);
    H_kw = (12.0 / (N_all*(N_all+1))) * ((sum(r_all(mask_c1))^2)/n_c1 + (sum(r_all(mask_c2))^2)/n_c2 + (sum(r_all(mask_c3))^2)/n_c3) - 3.0*(N_all+1);
    Triple_KW_Chi2(f_i) = max(0, H_kw); Triple_KW_PVal(f_i) = 1.0 - chi2cdf(Triple_KW_Chi2(f_i), 2);
end

AllTasks_MaxAUC(:, 4)  = Triple_Mean_OvR_AUC;
AllTasks_AbsCorr(:, 4) = Triple_Mean_AbsCorr;
AllTasks_AnovaF(:, 4)  = Triple_ANOVA_F;
AllTasks_MW_KW_Z(:, 4) = Triple_KW_Chi2;
AllTasks_KS_Stat(:, 4) = Triple_Max_KS;

Global_Mean_MaxAUC  = mean(AllTasks_MaxAUC(:, 1:3), 2);
Global_Best_MaxAUC  = max(AllTasks_MaxAUC(:, 1:3), [], 2);
Global_Mean_AbsCorr = mean(AllTasks_AbsCorr(:, 1:3), 2);
Global_Mean_AnovaF  = mean(AllTasks_AnovaF, 2);
Global_Mean_MW_Z    = mean(AllTasks_MW_KW_Z(:, 1:3), 2);
Global_Mean_KS      = mean(AllTasks_KS_Stat(:, 1:3), 2);

Global_Consensus_Score = 0.25*norm_01(Global_Mean_MaxAUC) + 0.15*norm_01(Global_Best_MaxAUC) + ...
                         0.15*norm_01(Global_Mean_AbsCorr) + 0.15*norm_01(Global_Mean_AnovaF) + ...
                         0.15*norm_01(Global_Mean_MW_Z) + 0.15*norm_01(Global_Mean_KS);

T_Global = table(Feature_Names_2880, Channel_List_2880, Domain_List_2880, CalType_List_2880, ...
    AllTasks_MaxAUC(:, 1), AllTasks_MaxAUC(:, 2), AllTasks_MaxAUC(:, 3), ...
    Global_Mean_MaxAUC, Global_Best_MaxAUC, Global_Mean_AbsCorr, Global_Mean_AnovaF, Triple_ANOVA_PVal, ...
    Global_Mean_MW_Z, Triple_KW_PVal, Global_Mean_KS, Global_Consensus_Score, ...
    'VariableNames', {'Full_Feature_Name', 'Channel', 'Domain', 'Calibration_Type', ...
    'MaxAUC_Amu_vs_Con', 'MaxAUC_Amu_vs_Fea', 'MaxAUC_Con_vs_Fea', ...
    'Global_Mean_MaxAUC', 'Global_Peak_MaxAUC', 'Global_Mean_Abs_Pearson', 'Global_Mean_ANOVA_F', 'Triple_ANOVA_P_Value', ...
    'Global_Mean_MannWhitney_Z', 'Triple_Kruskal_P_Value', 'Global_Mean_KS_Stat', 'Master_Consensus_Score'});

T_Master_Top100 = write_top_n_sheet(T_Global, 'Master_Consensus_Score', TOP_N, file_master_cal, 'Summary_Consensus_Top100');

[uniq_cal, ~, ic_cal] = unique(T_Master_Top100.Calibration_Type);
[cal_counts_sorted, idx_cal_s] = sort(accumarray(ic_cal, 1), 'descend');
T_Cal_Breakdown = table(uniq_cal(idx_cal_s), cal_counts_sorted, 'VariableNames', {'Calibration_Type', 'Count_In_Top100'});
writetable(T_Cal_Breakdown, file_master_cal, 'Sheet', 'CalibrationType_Breakdown');

[uniq_dom, ~, ic_dom] = unique(T_Master_Top100.Domain);
[dom_counts_sorted, idx_dom_s] = sort(accumarray(ic_dom, 1), 'descend');
T_Dom_Breakdown = table(uniq_dom(idx_dom_s), dom_counts_sorted, 'VariableNames', {'Feature_Domain', 'Count_In_Top100'});
writetable(T_Dom_Breakdown, file_master_cal, 'Sheet', 'Domain_Breakdown');

%% 7. PLOT & EXPORT THE 4-PANEL DIAGNOSTIC FIGURE FOR STEP 1
fprintf('\nGenerating Step 1 Calibrated Diagnostic Figure...\n');

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0.5, 0.5, 13.5, 9.5], 'Visible', 'on');

% Panel 1: Breakdown by Calibration Type (Spatial vs Asymmetry vs Subject-Calibrated)
ax1 = subplot(2, 2, 1);
cal_labels = T_Cal_Breakdown.Calibration_Type;
cal_vals   = T_Cal_Breakdown.Count_In_Top100;
barh(ax1, flipud(cal_vals), 0.55, 'FaceColor', [0.17, 0.36, 0.56], 'EdgeColor', 'k');
set(ax1, 'YTick', 1:length(cal_labels), 'YTickLabel', flipud(strrep(cal_labels, '_', '\_')), ...
    'FontSize', 8, 'FontWeight', 'normal');
xlabel(ax1, 'Number of Features in Global Top 100', 'FontSize', 8, 'FontWeight', 'normal');
ylabel(ax1, 'Calibration Method', 'FontSize', 8, 'FontWeight', 'normal');
title(ax1, 'Top 100 Features by Calibration Method', 'FontSize', 9, 'FontWeight', 'normal');
xlim(ax1, [0, max(cal_vals) + 8]);
grid(ax1, 'on');
vals_f = flipud(cal_vals);
for i = 1:length(vals_f)
    text(ax1, vals_f(i) + 0.8, i, sprintf('%d%%', vals_f(i)), 'FontSize', 8, 'VerticalAlignment', 'middle');
end

% Panel 2: Breakdown by Extraction Domain (Time, Freq, DB4, COIF5)
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
title(ax2, 'Top 100 Calibrated Features by Domain', 'FontSize', 9, 'FontWeight', 'normal');
ylim(ax2, [0, max(dom_vals) + 8]);
grid(ax2, 'on');
for i = 1:length(dom_vals)
    text(ax2, i, dom_vals(i) + 1.5, sprintf('%d (%d%%)', dom_vals(i), dom_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Panel 3: Top 15 Univariate Feature AUCs per Pair (with 0.574 & 0.627 Benchmarks)
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
plot(ax3, [0.5, 3.5], [0.574, 0.574], 'r--', 'LineWidth', 1.1);
text(ax3, 0.55, 0.576, 'Old Triad Best (0.574)', 'Color', 'r', 'FontSize', 7.5);
plot(ax3, [0.5, 3.5], [0.627, 0.627], 'b:', 'LineWidth', 1.2);
text(ax3, 0.55, 0.629, 'Uncalibrated 2240 Best (0.627)', 'Color', 'b', 'FontSize', 7.5);
hold(ax3, 'off');
set(ax3, 'FontSize', 8, 'FontWeight', 'normal');
ylabel(ax3, 'Discriminative Max AUC', 'FontSize', 8, 'FontWeight', 'normal');
title(ax3, 'Top 15 Calibrated Univariate Feature AUCs per Pair', 'FontSize', 9, 'FontWeight', 'normal');
grid(ax3, 'on');

% Panel 4: Heatmap of Top 10 Calibrated Master Consensus Features
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
set(ax4, 'XTick', 1:3, 'XTickLabel', {'Amu vs Con', 'Amu vs Fea', 'Con vs Fea'}, ...
    'YTick', 1:10, 'YTickLabel', strrep(top10_names, '_', '\_'), ...
    'FontSize', 7, 'FontWeight', 'normal');
title(ax4, 'Top 10 Calibrated Consensus Features', 'FontSize', 9, 'FontWeight', 'normal');

mid_auc = (min(top10_aucs(:)) + max(top10_aucs(:))) / 2;
for r_i = 1:10
    for c_i = 1:3
        val_rc = top10_aucs(r_i, c_i);
        txt_col = 'k'; if val_rc < mid_auc, txt_col = 'w'; end
        text(ax4, c_i, r_i, sprintf('%.3f', val_rc), ...
            'HorizontalAlignment', 'center', 'Color', txt_col, 'FontSize', 8);
    end
end

fig_png = 'Step1_Calibrated_Diagnostic_Figure.png';
fig_pdf = 'Step1_Calibrated_Diagnostic_Figure.pdf';
exportgraphics(fig, fig_png, 'Resolution', 300);
exportgraphics(fig, fig_pdf, 'ContentType', 'vector');

elapsed_sec = toc;
fprintf('\n=========================================================================\n');
fprintf('STEP 1 COMPLETED IN %.2f SECONDS!\n', elapsed_sec);
fprintf('Top #1 Calibrated Feature: %s (Peak AUC = %.4f, Mean AUC = %.4f)\n', ...
    T_Master_Top100.Full_Feature_Name{1}, T_Master_Top100.Global_Peak_MaxAUC(1), ...
    T_Master_Top100.Global_Mean_MaxAUC(1));
fprintf('Saved Files:\n  1) %s\n  2) %s\n  3) %s\n', cal_mat_file, file_master_cal, fig_png);
fprintf('=========================================================================\n');

%% --- HELPER FUNCTIONS ---
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