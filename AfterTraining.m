clear; close all;
load('mnist.mat');
load('NET_MNIST_RESNET18_sgdm.mat');

%% Change Y into categorical
YTrainCat = categorical(YTrain);
YTestCat = categorical(YTest);

%% Plot confusion matrix
YPredicted = classify(net,XTest);
plotconfusion(YTestCat,YPredicted);

%% Read input images
rng shuffle
r = randi([1 1000],1,5);
I = cell(size(r,2),1);
label = categorical();

figure;
for i = 1:size(r,2)
    I{i} = XTest(:,:,:,r(i));
    label(i) = classify(net,I{i});
    subplot(1,size(r,2),i);
    imshow(I{i});
end

label