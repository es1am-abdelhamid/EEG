%% =========================================================================
% EEG DATA PREPROCESSING & STRICT SUBJECT-LEVEL CLASS BALANCING
% Target Emotions: Amusement (4), Confusion (11), fear (17)
% =========================================================================
clear; clc; close all;

% --- 1. Configuration Parameters ---
raw_data_folder = 'eeg_raw'; % Folder containing the .txt files
target_emotions = [4, 11, 17];
Fs = 256; % Sampling frequency (Hz)
num_channels = 14;

% Windowing parameters
window_sec  = 10; % 10 seconds per segment
overlap_sec = 5;  % 50% overlap (5 seconds step)
win_len  = window_sec * Fs;  % 2560 samples
step_len = overlap_sec * Fs; % 1280 samples

% Filter design: Butterworth Bandpass 0.5 - 45 Hz
[b_filt, a_filt] = butter(4, [0.5 45]/(Fs/2), 'bandpass');

% --- 2. Read Files and Extract Segments ---
fprintf('Scanning folder "%s" for target emotions...\n', raw_data_folder);
files = dir(fullfile(raw_data_folder, '*.txt'));

All_Segments = [];
All_EmotionIDs = [];
All_ParticipantIDs = [];

for i = 1:length(files)
    filename = files(i).name;
    
    % Parse filename: e.g., '1_17.0.txt' -> Participant=1, Emotion=17
    parsed = sscanf(filename, '%d_%d.0.txt');
    if isempty(parsed) || length(parsed) < 2
        continue;
    end
    
    p_id = parsed(1);
    e_id = parsed(2);
    
    % Only process files matching our target emotions
    if ismember(e_id, target_emotions)
        filepath = fullfile(raw_data_folder, filename);
        
        try
            % Read data (assuming rows=samples, cols=channels)
            data = readmatrix(filepath);
            
            % Check if orientation is Channels x Samples and transpose if needed
            if size(data, 2) > 14 && size(data, 1) <= 14
                data = data';
            end
            
            % Ensure we only take the first 14 channels
            if size(data, 2) >= num_channels
                data = data(:, 1:num_channels);
            else
                fprintf('Skipping %s: Not enough channels.\n', filename);
                continue;
            end
            
            % Remove DC level (Zero-mean) per channel
            data = data - mean(data, 1);
            
            % Apply Bandpass Filter (0.5 - 45 Hz)
            data_filt = filtfilt(b_filt, a_filt, data);
            
            % Segment the data (10 seconds window, 5 seconds step)
            num_samples = size(data_filt, 1);
            for start_idx = 1:step_len:(num_samples - win_len + 1)
                end_idx = start_idx + win_len - 1;
                segment = data_filt(start_idx:end_idx, :); % [2560 x 14]
                
                % Append to master arrays
                All_Segments = cat(3, All_Segments, segment);
                All_EmotionIDs = [All_EmotionIDs; e_id];
                All_ParticipantIDs = [All_ParticipantIDs; p_id];
            end
        catch
            fprintf('Error reading or processing file: %s\n', filename);
        end
    end
end

fprintf('\nExtraction complete. Total raw segments: %d\n', length(All_EmotionIDs));

% --- 3. STRICT SUBJECT-LEVEL BALANCING ---
fprintf('\nApplying Strict Subject-Level Balancing...\n');
unique_participants = unique(All_ParticipantIDs);
balanced_indices = [];
retained_participants = 0;

for p = 1:length(unique_participants)
    p_id = unique_participants(p);
    
    % Find indices for this participant across the 3 emotions
    idx_e1 = find(All_ParticipantIDs == p_id & All_EmotionIDs == target_emotions(1));
    idx_e2 = find(All_ParticipantIDs == p_id & All_EmotionIDs == target_emotions(2));
    idx_e3 = find(All_ParticipantIDs == p_id & All_EmotionIDs == target_emotions(3));
    
    count_e1 = length(idx_e1);
    count_e2 = length(idx_e2);
    count_e3 = length(idx_e3);
    
    % Intersection Check: If participant lacks ANY of the 3 emotions, exclude them!
    if count_e1 == 0 || count_e2 == 0 || count_e3 == 0
        fprintf('  [Excluded] Participant %d: Missing one or more emotions (Counts: %d, %d, %d).\n', ...
            p_id, count_e1, count_e2, count_e3);
        continue;
    end
    
    % Find the minimum number of segments this participant has across the 3 emotions
    min_seg_for_p = min([count_e1, count_e2, count_e3]);
    fprintf('  [Included] Participant %d: Keeping exactly %d segments per emotion.\n', p_id, min_seg_for_p);
    
    % Randomly sample exactly 'min_seg_for_p' segments from each emotion
    rng(42 + p_id); % Seed for reproducibility per participant
    idx_e1_shuff = idx_e1(randperm(count_e1));
    idx_e2_shuff = idx_e2(randperm(count_e2));
    idx_e3_shuff = idx_e3(randperm(count_e3));
    
    balanced_indices = [balanced_indices; ...
                        idx_e1_shuff(1:min_seg_for_p); ...
                        idx_e2_shuff(1:min_seg_for_p); ...
                        idx_e3_shuff(1:min_seg_for_p)];
                    
    retained_participants = retained_participants + 1;
end

% Shuffle the final dataset globally so classes and participants are mixed
rng(42);
balanced_indices = balanced_indices(randperm(length(balanced_indices)));

% Extract the perfectly balanced dataset
Segments_Balanced = All_Segments(:, :, balanced_indices);
EmotionID_Balanced = All_EmotionIDs(balanced_indices);
ParticipantID_Balanced = All_ParticipantIDs(balanced_indices);

% --- 4. Save to .mat File ---
out_filename = 'eeg_41711.mat';
fprintf('\nSaving strictly balanced dataset to "%s"...\n', out_filename);

save(out_filename, 'Segments_Balanced', 'EmotionID_Balanced', ...
     'ParticipantID_Balanced', 'target_emotions', 'Fs', 'window_sec', 'overlap_sec', '-v7.3');

fprintf('\nDONE!\n');
fprintf('Retained Participants: %d\n', retained_participants);
fprintf('Final dataset shape: [%d Time_Samples x %d Channels x %d Segments]\n', ...
    size(Segments_Balanced, 1), size(Segments_Balanced, 2), size(Segments_Balanced, 3));
fprintf('Class distribution:\n');
for i=1:3
    fprintf('  Emotion %d: %d segments\n', target_emotions(i), sum(EmotionID_Balanced == target_emotions(i)));
end