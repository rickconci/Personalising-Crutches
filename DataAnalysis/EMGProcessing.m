%% Variables
clear 
close all
clc

EMGfilepath = "C:\Personalized Crutch Project\MIH27\EMG\";
filepath = "C:\Personalized Crutch Project\MIH27\";
filename = "maxisometric_1";
numtrials = 14;
maxsteptime = 2; % (seconds) exclude steps longer than this

% Bandpass filter inputs
f_low = 30;
f_high = 500;
n = 1; % fourth order // bandpass filter and filtfilt both double the order

% Lowpass filter inputs
f_lowpass = 6;
m = 2; % fourth order // filtfilt doubles the order

%% Load and find Max Isometric
data = load(EMGfilepath + filename + ".mat");
frequency = data.Fs(1); 
channelnames = data.Channels;
channelnames = strtrim(extractBefore(string(channelnames(contains(string(channelnames), 'EMG') & ~contains(string(channelnames), 'imu'), :)), ':'));
numchannels = length(channelnames);

maxiso = zeros(1, numchannels);

for i = 1:numchannels
    emg = data.Data(i, :);  
    emg_filtered = FilterEMG(emg, frequency, f_high, f_low, f_lowpass, n, m);
    maxiso(i) = max(emg_filtered);
end

%% Find Max Percent Activation for each Trial
maxpercent = zeros(numtrials, numchannels);

for i = 1:numtrials
    data = load(EMGfilepath + "Trial_" + num2str(i) + ".mat"); % load data
    for j = 1:numchannels
        emg = data.Data(j, :); % isolate and rectify data 
        emg = FilterEMG(emg, frequency, f_high, f_low, f_lowpass, n, m);
        emg = emg(1, 1:1:2*60*round(frequency)); % trim to 2 minutes
        maximum = max(emg);
        maxpercent(i, j) = (maximum / maxiso(j)) * 100;
    end
end

%% Segment EMG

% Load Sync Times
syncs = readtable(filepath + 'Sync_Indices.xlsx');

% Load Step Indices
stepstruct = load(filepath + "step_indices.mat");
steps = stepstruct.steps;

segments_all = struct();  % master struct for all trials



% For each trial
for i = 1:numtrials
    % Load trial EMG data
    data = load(EMGfilepath + "Trial_" + num2str(i) + ".mat");

    % Get step times for this trial
    trialname = "Trial_" + num2str(i);
    if ~isfield(steps, trialname)
        warning("No step data for %s — skipping.", trialname);
        continue
    end

    % Convert to seconds if needed
    step_times = steps.(trialname);
    if max(step_times) > 1000 % likely in samples, not seconds
        step_times = step_times / frequency;
    end

    % Adjust to sync time
    step_times = step_times - table2array(syncs(i, 1));

    % Preallocate storage for this trial
    trial_segments = struct();
    trial_avg = struct();

    % Process each EMG channel
    for j = 1:numchannels
        emg = data.Data(j, table2array(syncs(i, 2))*frequency:end); 
        [segments, avg] = EMGProcess(emg, step_times, frequency, f_high, f_low, f_lowpass, n, m, trialname, channelnames(j), maxsteptime);

        % Save segmented EMG and averaged waveform for this channel
        chan = matlab.lang.makeValidName(channelnames(j)); % ensures valid struct field
        trial_segments.(chan) = segments;
        trial_avg.(chan) = avg;

    end

    % Store under trial name
    segments_all.(trialname).segments = trial_segments;
    segments_all.(trialname).average = trial_avg;

end

%% Plot EMG Data
for i = 1:numchannels
    channelname = channelnames(i); % get channel name
    figure(i)
    for j = 1:numtrials
        subplot(5, 3, j)
        hold on
        trialname = "Trial_" + num2str(j); % get trial name

        % get average and standard deviation
        segments = segments_all.(trialname).segments.(channelname);
        segments = cell2mat(segments');
        t = linspace(1, 100, size(segments, 2));
        avg = mean(segments, 1);
        stdv = std(segments, 0, 1);
        upper = avg + stdv;
        lower = avg - stdv;

        fill([t fliplr(t)], [upper fliplr(lower)], [0.6 0.6 0.6], 'EdgeColor', 'none','FaceAlpha', 0.3);  % light blue shading
   
        % Plot the mean line
        plot(t, avg, 'k', 'LineWidth', 2);
        
        % Make it look nice
        xlabel('Percent Gait Cycle')
        ylabel('Percent Activation');
        title(['Trial ' num2str(j)])
        ylim([0 maxiso(i)])
        hold off
    end
    sgtitle(channelname)
end


%% Supplemental Functions

%% Function: Filter EMG Data
function EMG_filtered = FilterEMG(EMGData, frequency, f_high, f_low, f_lowpass, n, m)
    % Bandpass filter
    [b, a] = butter(n, [f_low f_high] / (frequency / 2));
    EMG_bp = filtfilt(b, a, EMGData);

    % Rectify
    EMG_abs = abs(EMG_bp);

    % Lowpass filter (linear envelope)
    [b, a] = butter(m, f_lowpass / (frequency / 2));
    EMG_filtered = filtfilt(b, a, EMG_abs);
end

%% Function: Process EMG with Step Segmentation
function [segments, avg] = EMGProcess(EMGData, step_times, frequency, f_high, f_low, f_lowpass, n, m, trialname, channel, maxsteptime)
    
    % --- Filter EMG Data ---
    EMG_filtered = FilterEMG(EMGData, frequency, f_high, f_low, f_lowpass, n, m);

    % --- Segment EMG Data by Step Times ---
    numsamples = length(EMG_filtered);
    trialtime = numsamples / frequency; % convert to seconds
    num_segments = length(step_times) - 1;
    maxlength = frequency * maxsteptime; % maximum samples per step

    segments = cell(1, num_segments);

    for k = 1:num_segments
        start_sec = step_times(k);
        end_sec = step_times(k + 1);
        segment = segmentEMGData(EMG_filtered, start_sec, end_sec, numsamples, trialtime, maxlength);
        if ~isempty(segment)
            segments{k} = segment;
        end
    end

    % Remove empty segments
    segments = segments(~cellfun(@isempty, segments));

    if isempty(segments)
        avg = [];
        return
    end

    % --- Normalize Each Step to % Gait Cycle ---
    max_length = max(cellfun(@length, segments));
    steps_percent = linspace(0, 100, max_length);
    avg = zeros(size(steps_percent));

    for s = 1:length(segments)
        seg = segments{s};
        percent = linspace(0, 100, length(seg));
        seg_interp = interp1(percent, seg, steps_percent, 'linear', 'extrap');
        avg = avg + seg_interp;
        segments{s} = seg_interp;
    end

    % Average across all steps
    avg = avg / length(segments);

    % % Check with Plot
    % figure; plot(steps_percent, avg)
    % title(sprintf("%s - %s", trialname, channel))
    % xlabel('% Step'); ylabel('EMG (filtered)')
end

%% Function to Segment EMG Data
function segment = segmentEMGData(emg, start_sec, end_sec, numsamples, trialtime, maxlength)
    start_idx = round(start_sec * numsamples / trialtime); % Start index
    end_idx = round(end_sec * numsamples / trialtime);     % End index
    steptime = end_idx - start_idx;
    if end_idx <= length(emg) && steptime < maxlength
        segment = emg(start_idx:end_idx); % Store segment
    else
        segment = [];
    end
end