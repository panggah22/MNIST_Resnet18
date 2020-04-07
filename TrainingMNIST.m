clear;
close all;
%% Load MNIST dataset
load('mnist.mat');

%% Change Y into categorical
YTrainCat = categorical(YTrain);
YTestCat = categorical(YTest);
% res = resnet18();

%% Choose training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTestCat}, ...
    'ValidationFrequency', 10, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Forming the layers and connection based on resnet-18
lgraphreal;
net = trainNetwork(XTrain, YTrainCat, LG, options);

%% Saving the result into Matlab file
save('NET_MNIST_RESNET18_sgdm.mat', 'net');
