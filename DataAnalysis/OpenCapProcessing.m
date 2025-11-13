%% Variables
clear 
close all
clc

filepath = "C:\Personalized Crutch Project\MIH27\OpenCap Data\OpenSimData\Kinematics\";
% filename = "test_crutch";
anglename = "knee_angle_r"; % (knee_angle_r, knee_angle_l,...)
numtrials = 14;

%% Import Data

data = importdata(filepath + filename + ".mot");
angle_data = data.data;
angles = data.colheaders;

%% Plot Angle

angle = angle_data(:, find(angles==anglename));
time = angle_data(:, find(angles=="time"));
plottitle = strrep(anglename, "_", " ");

hold on
plot(time, angle)
title(plottitle)
xlim("tight")
xlabel("time (sec)")
ylim([-5 100])
ylabel("angle (degrees)")
hold off

%% Average Lumbar Extension

averages = table('Size', [numtrials 4], ...
    'VariableTypes', {'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Trial', 'Subtrial1', 'Subtrial2', 'Subtrial3'}); % create table to hold results

for i = 1:numtrials % iterate through trials
    averages.Trial(i) = i;

    for j = 1:3 % iterate through subtrials
        filename = filepath + "Trial_" + num2str(i) + "_" + num2str(j) + ".mot";

        if isfile(filename) % check if data exists
            % import data
            data = importdata(filename);
            angle_data = data.data;
            angles = data.colheaders;

            % extract lumbar extension
            lumbar_extenstion = angle_data(:, find(angles=="lumbar_extension"));
            time = angle_data(:, find(angles=="time"));

            % trim data
            trim_time = 1; % amount to trim from beginning and end (seconds)
            frequency = time(2);
            lumbar_extenstion = lumbar_extenstion(round(trim_time/frequency):round(end - trim_time/frequency), :);

            % calculate average angle
            avg_lumbar_extension = mean(lumbar_extenstion);

            % add angle to table
            averages.(sprintf('Subtrial%d', j))(i) = avg_lumbar_extension;

        else
            % if missing file, fill with NaN
            averages.(sprintf('Subtrial%d', j))(i) = NaN;
        end
    end
end

% write to Excel
output_file = fullfile(filepath, 'Average_Lumbar_Extension.xlsx');
writetable(averages, output_file);
