clear all
clc

format long

% Dataset of 105 Person
Dataset_105_C5 = load('Dataset_105_C5');
Dataset_105_C5 = Dataset_105_C5.Dataset;
Dataset_105_C5 = reshape(Dataset_105_C5, [8820, 640]);

Dataset_105_C3 = load('Dataset_105_C3');
Dataset_105_C3 = Dataset_105_C3.Dataset;
Dataset_105_C3 = reshape(Dataset_105_C3, [8820, 640]);

Dataset_105_C1 = load('Dataset_105_C1');
Dataset_105_C1 = Dataset_105_C1.Dataset;
Dataset_105_C1 = reshape(Dataset_105_C1, [8820, 640]);

Dataset_105_C2 = load('Dataset_105_C2');
Dataset_105_C2 = Dataset_105_C2.Dataset;
Dataset_105_C2 = reshape(Dataset_105_C2, [8820, 640]);

Dataset_105_C4 = load('Dataset_105_C4');
Dataset_105_C4 = Dataset_105_C4.Dataset;
Dataset_105_C4 = reshape(Dataset_105_C4, [8820, 640]);

Dataset_105_C6 = load('Dataset_105_C6');
Dataset_105_C6 = Dataset_105_C6.Dataset;
Dataset_105_C6 = reshape(Dataset_105_C6, [8820, 640]);

Dataset_105_CP5 = load('Dataset_105_CP5');
Dataset_105_CP5 = Dataset_105_CP5.Dataset;
Dataset_105_CP5 = reshape(Dataset_105_CP5, [8820, 640]);

Dataset_105_CP3 = load('Dataset_105_CP3');
Dataset_105_CP3 = Dataset_105_CP3.Dataset;
Dataset_105_CP3 = reshape(Dataset_105_CP3, [8820, 640]);

Dataset_105_CP1 = load('Dataset_105_CP1');
Dataset_105_CP1 = Dataset_105_CP1.Dataset;
Dataset_105_CP1 = reshape(Dataset_105_CP1, [8820, 640]);

Dataset_105_CP2 = load('Dataset_105_CP2');
Dataset_105_CP2 = Dataset_105_CP2.Dataset;
Dataset_105_CP2 = reshape(Dataset_105_CP2, [8820, 640]);

Dataset_105_CP4 = load('Dataset_105_CP4');
Dataset_105_CP4 = Dataset_105_CP4.Dataset;
Dataset_105_CP4 = reshape(Dataset_105_CP4, [8820, 640]);

Dataset_105_CP6 = load('Dataset_105_CP6');
Dataset_105_CP6 = Dataset_105_CP6.Dataset;
Dataset_105_CP6 = reshape(Dataset_105_CP6, [8820, 640]);

Dataset_105_P5 = load('Dataset_105_P5');
Dataset_105_P5 = Dataset_105_P5.Dataset;
Dataset_105_P5 = reshape(Dataset_105_P5, [8820, 640]);

Dataset_105_P3 = load('Dataset_105_P3');
Dataset_105_P3 = Dataset_105_P3.Dataset;
Dataset_105_P3 = reshape(Dataset_105_P3, [8820, 640]);

Dataset_105_P1 = load('Dataset_105_P1');
Dataset_105_P1 = Dataset_105_P1.Dataset;
Dataset_105_P1 = reshape(Dataset_105_P1, [8820, 640]);

Dataset_105_P2 = load('Dataset_105_P2');
Dataset_105_P2 = Dataset_105_P2.Dataset;
Dataset_105_P2 = reshape(Dataset_105_P2, [8820, 640]);

Dataset_105_P4 = load('Dataset_105_P4');
Dataset_105_P4 = Dataset_105_P4.Dataset;
Dataset_105_P4 = reshape(Dataset_105_P4, [8820, 640]);

Samples = vertcat(Dataset_105_C5, Dataset_105_C3, Dataset_105_C1, Dataset_105_C2,...
    Dataset_105_C4, Dataset_105_C6, Dataset_105_CP5, Dataset_105_CP3,...
    Dataset_105_CP1, Dataset_105_CP2, Dataset_105_CP4, Dataset_105_CP6,...
    Dataset_105_P5, Dataset_105_P3, Dataset_105_P1, Dataset_105_P2,...
    Dataset_105_P4);

[m, n] = size(Samples);
for i = 1:n
    mean_x = mean(Samples(:, i));
    std_x  = std(Samples(:, i));
    Samples(:, i) = (Samples(:, i)- mean_x) / std_x;
end

%%
% Labels of 105 Person
Labels_105_C5 = load('Labels_105_C5');
Labels_105_C5 = Labels_105_C5.Labels;
Labels_105_C5 = reshape(Labels_105_C5, [8820, 4]);

Labels_105_C3 = load('Labels_105_C3');
Labels_105_C3 = Labels_105_C3.Labels;
Labels_105_C3 = reshape(Labels_105_C3, [8820, 4]);

Labels_105_C1 = load('Labels_105_C1');
Labels_105_C1 = Labels_105_C1.Labels;
Labels_105_C1 = reshape(Labels_105_C1, [8820, 4]);

Labels_105_C2 = load('Labels_105_C2');
Labels_105_C2 = Labels_105_C2.Labels;
Labels_105_C2 = reshape(Labels_105_C2, [8820, 4]);

Labels_105_C4 = load('Labels_105_C4');
Labels_105_C4 = Labels_105_C4.Labels;
Labels_105_C4 = reshape(Labels_105_C4, [8820, 4]);

Labels_105_C6 = load('Labels_105_C6');
Labels_105_C6 = Labels_105_C6.Labels;
Labels_105_C6 = reshape(Labels_105_C6, [8820, 4]);

Labels_105_CP5 = load('Labels_105_CP5');
Labels_105_CP5 = Labels_105_CP5.Labels;
Labels_105_CP5 = reshape(Labels_105_CP5, [8820, 4]);

Labels_105_CP3 = load('Labels_105_CP3');
Labels_105_CP3 = Labels_105_CP3.Labels;
Labels_105_CP3 = reshape(Labels_105_CP3, [8820, 4]);

Labels_105_CP1 = load('Labels_105_CP1');
Labels_105_CP1 = Labels_105_CP1.Labels;
Labels_105_CP1 = reshape(Labels_105_CP1, [8820, 4]);

Labels_105_CP2 = load('Labels_105_CP2');
Labels_105_CP2 = Labels_105_CP2.Labels;
Labels_105_CP2 = reshape(Labels_105_CP2, [8820, 4]);

Labels_105_CP4 = load('Labels_105_CP4');
Labels_105_CP4 = Labels_105_CP4.Labels;
Labels_105_CP4 = reshape(Labels_105_CP4, [8820, 4]);

Labels_105_CP6 = load('Labels_105_CP6');
Labels_105_CP6 = Labels_105_CP6.Labels;
Labels_105_CP6 = reshape(Labels_105_CP6, [8820, 4]);

Labels_105_P5 = load('Labels_105_P5');
Labels_105_P5 = Labels_105_P5.Labels;
Labels_105_P5 = reshape(Labels_105_P5, [8820, 4]);

Labels_105_P3 = load('Labels_105_P3');
Labels_105_P3 = Labels_105_P3.Labels;
Labels_105_P3 = reshape(Labels_105_P3, [8820, 4]);

Labels_105_P1 = load('Labels_105_P1');
Labels_105_P1 = Labels_105_P1.Labels;
Labels_105_P1 = reshape(Labels_105_P1, [8820, 4]);

Labels_105_P2 = load('Labels_105_P2');
Labels_105_P2 = Labels_105_P2.Labels;
Labels_105_P2 = reshape(Labels_105_P2, [8820, 4]);

Labels_105_P4 = load('Labels_105_P4');
Labels_105_P4 = Labels_105_P4.Labels;
Labels_105_P4 = reshape(Labels_105_P4, [8820, 4]);

Labels = vertcat(Labels_105_C5, Labels_105_C3, Labels_105_C1, Labels_105_C2,...
    Labels_105_C4, Labels_105_C6, Labels_105_CP5, Labels_105_CP3,...
    Labels_105_CP1, Labels_105_CP2, Labels_105_CP4, Labels_105_CP6,...
    Labels_105_P5, Labels_105_P3, Labels_105_P1, Labels_105_P2,...
    Labels_105_P4);

%%
% Shuffle the Overall Data
Dataset = [Samples, Labels];
[m, n] = find(isnan(Dataset));
Dataset(m, :) = [];
rowrank = randperm(size(Dataset, 1)); 
Dataset = Dataset(rowrank, :); 

%%
Training_data_1 = Dataset(1:67243, 1:640);
xlswrite('Training_data_1.xlsx', Training_data_1);

%%
Training_data_2 = Dataset(67244:134487, 1:640);
xlswrite('Training_data_2.xlsx', Training_data_2);

%%
Training_labels = Dataset(1:134487, 641:644);
xlswrite('Training_labels.xlsx', Training_labels);

%%
% % Save Data to Excel File 
Test_data = Dataset(134488:149430, 1:640);
Test_labels = Dataset(134488:149430, 641:644);
xlswrite('Test_data.xlsx', Test_data);
xlswrite('Test_labels.xlsx', Test_labels);

%%
clear all
clc

format long

% Dataset of 4 Person
Dataset_4_C5 = load('Dataset_4_C5');
Dataset_4_C5 = Dataset_4_C5.Dataset;
Dataset_4_C5 = reshape(Dataset_4_C5, [336, 640]);

Dataset_4_C3 = load('Dataset_4_C3');
Dataset_4_C3 = Dataset_4_C3.Dataset;
Dataset_4_C3 = reshape(Dataset_4_C3, [336, 640]);

Dataset_4_C1 = load('Dataset_4_C1');
Dataset_4_C1 = Dataset_4_C1.Dataset;
Dataset_4_C1 = reshape(Dataset_4_C1, [336, 640]);

Dataset_4_C2 = load('Dataset_4_C2');
Dataset_4_C2 = Dataset_4_C2.Dataset;
Dataset_4_C2 = reshape(Dataset_4_C2, [336, 640]);

Dataset_4_C4 = load('Dataset_4_C4');
Dataset_4_C4 = Dataset_4_C4.Dataset;
Dataset_4_C4 = reshape(Dataset_4_C4, [336, 640]);

Dataset_4_C6 = load('Dataset_4_C6');
Dataset_4_C6 = Dataset_4_C6.Dataset;
Dataset_4_C6 = reshape(Dataset_4_C6, [336, 640]);

Dataset_4_CP5 = load('Dataset_4_CP5');
Dataset_4_CP5 = Dataset_4_CP5.Dataset;
Dataset_4_CP5 = reshape(Dataset_4_CP5, [336, 640]);

Dataset_4_CP3 = load('Dataset_4_CP3');
Dataset_4_CP3 = Dataset_4_CP3.Dataset;
Dataset_4_CP3 = reshape(Dataset_4_CP3, [336, 640]);

Dataset_4_CP1 = load('Dataset_4_CP1');
Dataset_4_CP1 = Dataset_4_CP1.Dataset;
Dataset_4_CP1 = reshape(Dataset_4_CP1, [336, 640]);

Dataset_4_CP2 = load('Dataset_4_CP2');
Dataset_4_CP2 = Dataset_4_CP2.Dataset;
Dataset_4_CP2 = reshape(Dataset_4_CP2, [336, 640]);

Dataset_4_CP4 = load('Dataset_4_CP4');
Dataset_4_CP4 = Dataset_4_CP4.Dataset;
Dataset_4_CP4 = reshape(Dataset_4_CP4, [336, 640]);

Dataset_4_CP6 = load('Dataset_4_CP6');
Dataset_4_CP6 = Dataset_4_CP6.Dataset;
Dataset_4_CP6 = reshape(Dataset_4_CP6, [336, 640]);

Dataset_4_P5 = load('Dataset_4_P5');
Dataset_4_P5 = Dataset_4_P5.Dataset;
Dataset_4_P5 = reshape(Dataset_4_P5, [336, 640]);

Dataset_4_P3 = load('Dataset_4_P3');
Dataset_4_P3 = Dataset_4_P3.Dataset;
Dataset_4_P3 = reshape(Dataset_4_P3, [336, 640]);

Dataset_4_P1 = load('Dataset_4_P1');
Dataset_4_P1 = Dataset_4_P1.Dataset;
Dataset_4_P1 = reshape(Dataset_4_P1, [336, 640]);

Dataset_4_P2 = load('Dataset_4_P2');
Dataset_4_P2 = Dataset_4_P2.Dataset;
Dataset_4_P2 = reshape(Dataset_4_P2, [336, 640]);

Dataset_4_P4 = load('Dataset_4_P4');
Dataset_4_P4 = Dataset_4_P4.Dataset;
Dataset_4_P4 = reshape(Dataset_4_P4, [336, 640]);

Samples = vertcat(Dataset_4_C5, Dataset_4_C3, Dataset_4_C1, Dataset_4_C2,...
    Dataset_4_C4, Dataset_4_C6, Dataset_4_CP5, Dataset_4_CP3,...
    Dataset_4_CP1, Dataset_4_CP2, Dataset_4_CP4, Dataset_4_CP6,...
    Dataset_4_P5, Dataset_4_P3, Dataset_4_P1, Dataset_4_P2,...
    Dataset_4_P4);

[m, n] = size(Samples);
for i = 1:n
    mean_x = mean(Samples(:, i));
    std_x  = std(Samples(:, i));
    Samples(:, i) = (Samples(:, i)- mean_x) / std_x;
end

%%
% Labels of 4 Person
Labels_4_C5 = load('Labels_4_C5');
Labels_4_C5 = Labels_4_C5.Labels;
Labels_4_C5 = reshape(Labels_4_C5, [336, 4]);

Labels_4_C3 = load('Labels_4_C3');
Labels_4_C3 = Labels_4_C3.Labels;
Labels_4_C3 = reshape(Labels_4_C3, [336, 4]);

Labels_4_C1 = load('Labels_4_C1');
Labels_4_C1 = Labels_4_C1.Labels;
Labels_4_C1 = reshape(Labels_4_C1, [336, 4]);

Labels_4_C2 = load('Labels_4_C2');
Labels_4_C2 = Labels_4_C2.Labels;
Labels_4_C2 = reshape(Labels_4_C2, [336, 4]);

Labels_4_C4 = load('Labels_4_C4');
Labels_4_C4 = Labels_4_C4.Labels;
Labels_4_C4 = reshape(Labels_4_C4, [336, 4]);

Labels_4_C6 = load('Labels_4_C6');
Labels_4_C6 = Labels_4_C6.Labels;
Labels_4_C6 = reshape(Labels_4_C6, [336, 4]);

Labels_4_CP5 = load('Labels_4_CP5');
Labels_4_CP5 = Labels_4_CP5.Labels;
Labels_4_CP5 = reshape(Labels_4_CP5, [336, 4]);

Labels_4_CP3 = load('Labels_4_CP3');
Labels_4_CP3 = Labels_4_CP3.Labels;
Labels_4_CP3 = reshape(Labels_4_CP3, [336, 4]);

Labels_4_CP1 = load('Labels_4_CP1');
Labels_4_CP1 = Labels_4_CP1.Labels;
Labels_4_CP1 = reshape(Labels_4_CP1, [336, 4]);

Labels_4_CP2 = load('Labels_4_CP2');
Labels_4_CP2 = Labels_4_CP2.Labels;
Labels_4_CP2 = reshape(Labels_4_CP2, [336, 4]);

Labels_4_CP4 = load('Labels_4_CP4');
Labels_4_CP4 = Labels_4_CP4.Labels;
Labels_4_CP4 = reshape(Labels_4_CP4, [336, 4]);

Labels_4_CP6 = load('Labels_4_CP6');
Labels_4_CP6 = Labels_4_CP6.Labels;
Labels_4_CP6 = reshape(Labels_4_CP6, [336, 4]);

Labels_4_P5 = load('Labels_4_P5');
Labels_4_P5 = Labels_4_P5.Labels;
Labels_4_P5 = reshape(Labels_4_P5, [336, 4]);

Labels_4_P3 = load('Labels_4_P3');
Labels_4_P3 = Labels_4_P3.Labels;
Labels_4_P3 = reshape(Labels_4_P3, [336, 4]);

Labels_4_P1 = load('Labels_4_P1');
Labels_4_P1 = Labels_4_P1.Labels;
Labels_4_P1 = reshape(Labels_4_P1, [336, 4]);

Labels_4_P2 = load('Labels_4_P2');
Labels_4_P2 = Labels_4_P2.Labels;
Labels_4_P2 = reshape(Labels_4_P2, [336, 4]);

Labels_4_P4 = load('Labels_4_P4');
Labels_4_P4 = Labels_4_P4.Labels;
Labels_4_P4 = reshape(Labels_4_P4, [336, 4]);

Labels = vertcat(Labels_4_C5, Labels_4_C3, Labels_4_C1, Labels_4_C2,...
    Labels_4_C4, Labels_4_C6, Labels_4_CP5, Labels_4_CP3,...
    Labels_4_CP1, Labels_4_CP2, Labels_4_CP4, Labels_4_CP6,...
    Labels_4_P5, Labels_4_P3, Labels_4_P1, Labels_4_P2,...
    Labels_4_P4);

%%
% Shuffle the Overall Data
Dataset = [Samples, Labels];
[m, n] = find(isnan(Dataset));
Dataset(m, :) = [];
rowrank = randperm(size(Dataset, 1)); 
Dataset = Dataset(rowrank, :); 

%%
% % Save Data to Excel File 
Validation_data = Dataset(1:5712, 1:640);
Validation_labels = Dataset(1:5712, 641:644);
xlswrite('Validation_data.xlsx', Validation_data);
xlswrite('Validation_labels.xlsx', Validation_labels);

