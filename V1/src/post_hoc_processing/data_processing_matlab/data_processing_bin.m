function data = data_processing_bin(fileName)

% --- Define the number of data fields and their data types
% --- fieldTypes = {'uint32', 'single'}; % timeMsec (uint32), load (float)

%%
% Open the binary file for reading
fileID = fopen(fileName,'r');

% Read the binary data from the file
data_all = fread(fileID, Inf, 'uint8=>uint8');
    
numFields = countElementsBetweenFirstPairs(data_all, [170 170], [187 187]);
fieldTypes = repmat({'single'}, 1, numFields); % Create a cell array with 'numFields' copies of 'single'

% Compute the number of bytes for each field type
fieldBytes = zeros(1, numel(fieldTypes));
for i = 1:numel(fieldTypes)
    % Create a temporary variable with the data type and a value of 0,
    % and compute the number of bytes in the resulting array
    tmp = typecast(eval([fieldTypes{i} '(0)']), 'uint8');
    fieldBytes(i) = numel(tmp);
end
% numFields = length(fieldTypes);

% Compute the byte offsets for each field in the data structure
fieldOffsets = [3, cumsum(fieldBytes(1:end-1)) + 3];

% Define the number of bytes per sample and the filename
numbBytes = 4 + sum(fieldBytes); % 4 is padding bytes

% Define the names of the data fields
fieldNames = cell(1, numFields);
for i = 1:numFields
    fieldNames{i} = sprintf('data%d', i);
end

% Create the data structure with empty arrays for each field
data = struct;
for i = 1:numFields
    data.(fieldNames{i}) = [];
end

% Close the binary file
fclose(fileID);

% Compute the number of samples
numb_sample = floor(length(data_all)/numbBytes);

% Loop over each sample and extract the data values for each field
for i = 1:numb_sample
    % Check the validity of the data packet
    checkDataPacket(data_all, i, numbBytes);

    % Loop over each field in the data structure
    for j = 1:numFields
        % Compute the byte range for the current field
        byteStart = (i-1)*numbBytes + fieldOffsets(j);
        byteEnd = byteStart + fieldBytes(j) - 1;

        % Extract the data value for the current field
        data.(sprintf('data%d',j))(i, 1) = typecast(uint8(data_all(byteStart:byteEnd)), fieldTypes{j});
    end
end
end

% Check the validity of the data packet
function checkDataPacket(data_all, i, numbBytes)
if (i-1)*numbBytes+numbBytes > length(data_all)
    error('Data packet %d is invalid', i);
end
end

function numFields = countElementsBetweenFirstPairs(data, pair1, pair2)
    % Initialize indices to -1 (indicating not found)
    startIndex = -1;
    endIndex = -1;

    % Find the first occurrence of pair1
    for i = 1:(length(data) - 1)
        if data(i) == pair1(1) && data(i+1) == pair1(2)
            startIndex = i + 1;  % Plus 1 because we want the index after the pair
            break;
        end
    end

    % If pair1 is found, find the first occurrence of pair2 after startIndex
    if startIndex ~= -1
        for j = (startIndex + 1):(length(data) - 1)
            if data(j) == pair2(1) && data(j+1) == pair2(2)
                endIndex = j;  % This is the start of the second pair
                break;
            end
        end
    end

    % Check if both pairs are found and end pair comes after start pair
    if startIndex ~= -1 && endIndex ~= -1
        % Count elements between the pairs
        count = endIndex - startIndex - 1;
    else
        error('One or both pairs not found in the list, or end pair occurs before start pair.');
    end

    numFields = count/4;
end