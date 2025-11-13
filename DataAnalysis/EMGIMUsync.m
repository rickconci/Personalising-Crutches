clear 
clc
close all

%% Variables

imufilepath = "C:\Personalized Crutch Project\MIH27\IMU_LoadCell\";
emgfilepath = "C:\Personalized Crutch Project\MIH27\EMG\";
filepath = "C:\Personalized Crutch Project\MIH27\";
fileprefix = "Trial_";
lcimu_fs = 500;  % Load Cell IMU sampling frequency (Hz)
numtrials = 14;

%% Load Data

peakindices = zeros(numtrials, 2); % initialize array to store peak locations. Row 1 from Load Cell. Row 2 from EMG

for i = 1:numtrials
    
    % load data 
    loadcellfile = imufilepath + fileprefix + num2str(i) + ".csv";
    emgfile = emgfilepath + fileprefix + num2str(i) + ".mat";
    
    loadcelldata = readtable(loadcellfile);
    emgdata = load(emgfile); 
    
    % Process IMU data
    t_lcimu = loadcelldata.acc_x_time / lcimu_fs;
    z_lcimu = loadcelldata.acc_z_data;
    
    % Smooth and find first significant peak
    z_smooth = movmean(z_lcimu, 100);
    [pks_lcimu, locs_lcimu] = findpeaks(z_smooth, 'MinPeakProminence', 1);
    
    
    lcimu_peak_idx = locs_lcimu(1);
    peakindices(i, 1) = locs_lcimu(1)/lcimu_fs; % store peak loc
    lcimu_peak_time = t_lcimu(lcimu_peak_idx);
    
    % Process EMG IMU data
    emgimu_fs = emgdata.Fs(10); % get sampling frequncy (Hz)
    t_emgimu = emgdata.Time(10, :);      % get time vector
    z_emgimu = emgdata.Data(10, :) * -10;  % get and scale z data
    
    % Smooth and find first significant EMG peak
    emg_smooth = movmean(z_emgimu, 100);  
    [pks_emgimu, locs_emgimu] = findpeaks(emg_smooth, 'MinPeakProminence', 1);
    
    
    emgimu_peak_idx = locs_emgimu(1);
    peakindices(i, 2) = locs_emgimu(1)/emgimu_fs; % store peak loc
    emgimu_peak_time = t_emgimu(emgimu_peak_idx);
    
    % Align IMU data
    offset = emgimu_peak_time - lcimu_peak_time;
    
    % Align by shifting EMG time vector so peaks coincide
    t_emgimu_aligned = t_emgimu - offset;
    
    % % Check with Plot
    % figure(i)
    % hold on
    % plot(t_lcimu, z_lcimu, 'b')
    % plot(t_emgimu_aligned, z_emgimu, 'r')
    % xline(lcimu_peak_time, 'k--', 'Aligned Peak')
    % legend('IMU (load cell)', 'IMU (EMG)', 'Alignment point')
    % xlabel('Time (s)')
    % ylabel('Signal')
    % title(sprintf('Trial %d — IMUs Aligned by First Peaks', i))
    % xlim([lcimu_peak_time - 5, lcimu_peak_time + 20])
end

%% Find Load Cells Steps

steps = []; % initialize struct to hold steps

for k = 1:numtrials
    % load trial
    loadcellfile = imufilepath + fileprefix + num2str(k) + ".csv";
    loadcelldata = readtable(loadcellfile);

    % look for times where foot is stationary
    force = loadcelldata.force;    
    time = loadcelldata.acc_x_time;
           
    % create time window and calculate threshold
    time_window = 0.05*lcimu_fs; % check 0.05 second intervals
    forcethreshold = mean(force);
    
    % find window starts and ends for foot
    windows = NaN(length(force),2); % first column for start times, second for end times
    i = 1; % indexing through frames
    j = 1; % count identified windows
    
    % get stride windows
    while i < length(force) - time_window
        if ((force(i) > forcethreshold) && (force(i+time_window) > forcethreshold))
            windows(j,1) = i; % set start time
            while force(i+time_window) > forcethreshold && (i < length(force) - time_window)
                i = i+1; % index through frames until force falls below threshold
            end
            i = i + time_window - 1; % go to end of window
            windows(j,2) = i; % set end time
            j = j+1; % next window
        else 
            i = i+1; % next frame
        end
    end
    
    windows = windows(~isnan(windows(:, 1)), :); % remove empty
    
    % check that windows aren't back to back
    window_count = 2;
    for i=2:length(windows(:,1)) - 1
        if window_count >= length(windows(:,1))
            break
        elseif windows(window_count, 1) - windows(window_count - 1, 2) < 5
            windows(window_count,1) = NaN;
            windows(window_count,2) = NaN;
            windows = windows(~isnan(windows(:, 1)), :); % remove row
        else
            window_count = window_count + 1;
        end
    end

    steps.(['Trial_' num2str(k)]) = windows(:, 1)*5/lcimu_fs;
    
    % % Check with Plot
    % figure(k)
    % 
    % % get start and end point values
    % s_points = force(windows(:, 1));
    % e_points = force(windows(:, 2));
    % 
    % % plot 
    % figure(k);
    % title(sprintf('Trial %d — Check Steps Graph', k))
    % hold on
    % plot(time, force)
    % scatter(windows(:, 1)*5, s_points)
    % scatter(windows(:, 2)*5, e_points)
    % legend('force', 'start', 'end')
    % axis('tight')
    % xlim([0 10000])
    % hold off
end

%% Save Data

% write Sync Points to Excel
output_file = fullfile(filepath, 'Sync_Indices.xlsx');
writematrix(peakindices, output_file); % column 1: load cell, column 2 EMG

% write Step Indices to .mat
output_file = fullfile(filepath, 'Step_Indices.mat');
save(output_file, 'steps');

