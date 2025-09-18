function metList = data_processing_metabolics(fileName)

estimateThresholdTime = 4.8; % [min]

weight = 77; % [kg]

averageTime = 2; % [min]
tau = 42;

figure(); hold on;
xlabel('Time (s)'); ylabel('Metabolic Cost (W kg^{-1}))');
default_color();
% legend('Location','best');


T = readcell(strcat(fileName,'.xlsx'));
expTime = datestr(T{2,5}, 'HH:MM:SS AM');
if expTime(end-1:end) == 'AM'
    expTime(1:3) = [];
else
    expTime(6:8) = [];
    if (str2num(expTime(1:2)) < 12)
        expTime(1:2) = num2str(str2num(expTime(1:2)) + 12);
    end
end
expTime(6:8) = [];
T = T(:,10:end);

% Find the index of the columns containing EEm, VO2, and VCO2
time_index = find(strcmp(T(1,:),'t'));
VO2_index = find(strcmp(T(1,:),'VO2'));
VCO2_index = find(strcmp(T(1,:),'VCO2'));
marker_index = find(strcmp(T(1,:),'Marker'));

time = cell2mat(T(4:end, time_index))*86400; % [s] 1 day = 86400 seconds

time = time - time(1);

[~, cut_idx] = min(abs(time-60*5));
T(cut_idx+4:end, :) = [];
time(cut_idx+1:end) = [];

VO2 = cell2mat(T(4:end, VO2_index)); % [mL/min]
VCO2 = cell2mat(T(4:end, VCO2_index)); % [mL/min]

y_meas = (0.278*VO2 + 0.075*VCO2)/weight; % [W/kg]

if time(end) < 60*estimateThresholdTime
    [y_average, y_bar, ~] = metabolic_rate_estimation(time,y_meas,tau);
else
    startTime = time(end) - 60 * averageTime;
    [~, startTimeIdx] = min(abs(time - startTime));

    y_average = mean(y_meas(startTimeIdx:end));

    [~, endTimeIdx] = min(abs(time - 180));
    time_estimate = time(1:endTimeIdx);
    [y_estimate, y_bar, ~] = metabolic_rate_estimation(time_estimate,y_meas(1:endTimeIdx),tau);
end

fprintf('Average: (%s, %ds) %s = %.4f W/kg\n', expTime, round(time(end)), fileName, y_average);

if time(end) >= 60*estimateThresholdTime
    fprintf('Estimtae: (%s, %ds) %s (est.) = %.4f W/kg\n', expTime, round(time(end)), fileName, y_estimate);
end

metList = y_average;

plot(time, y_meas, 'o');

if time(end) < 60 * estimateThresholdTime
    plot(time, y_bar, '-', 'HandleVisibility', 'off');
else    
    plot(time_estimate, y_bar, '-', 'HandleVisibility', 'off');

    xline(time(startTimeIdx), 'HandleVisibility', 'off');
    yline(y_average, 'HandleVisibility', 'off');

end

end
