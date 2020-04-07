LG = layerGraph();

tempLayers = [
    imageInputLayer([28 28 1],"Name","data")
    convolution2dLayer([4 4],64,"Name","conv1","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2b_branch2a")
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2b_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3a_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b_branch2a")
    reluLayer("Name","res3b_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3b")
    reluLayer("Name","res3b_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a")
    reluLayer("Name","res4a_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4a_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b_branch2a")
    reluLayer("Name","res4b_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4b")
    reluLayer("Name","res4b_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a")
    reluLayer("Name","res5a_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5a_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5b_branch2a")
    reluLayer("Name","res5b_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5b_branch2b")];
LG = addLayers(LG,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5b")
    reluLayer("Name","res5b_relu")
    globalAveragePooling2dLayer("Name","pool5")
    fullyConnectedLayer(10,"Name","fc10")
    softmaxLayer("Name","prob")
    classificationLayer("Name","ClassificationLayer_predictions")];
LG = addLayers(LG,tempLayers);

% clean up helper variable
clear tempLayers;

LG = connectLayers(LG,"pool1","res2a_branch2a");
LG = connectLayers(LG,"pool1","res2a/in2");
LG = connectLayers(LG,"bn2a_branch2b","res2a/in1");
LG = connectLayers(LG,"res2a_relu","res2b_branch2a");
LG = connectLayers(LG,"res2a_relu","res2b/in2");
LG = connectLayers(LG,"bn2b_branch2b","res2b/in1");
LG = connectLayers(LG,"res2b_relu","res3a_branch2a");
LG = connectLayers(LG,"res2b_relu","res3a_branch1");
LG = connectLayers(LG,"bn3a_branch1","res3a/in2");
LG = connectLayers(LG,"bn3a_branch2b","res3a/in1");
LG = connectLayers(LG,"res3a_relu","res3b_branch2a");
LG = connectLayers(LG,"res3a_relu","res3b/in2");
LG = connectLayers(LG,"bn3b_branch2b","res3b/in1");
LG = connectLayers(LG,"res3b_relu","res4a_branch2a");
LG = connectLayers(LG,"res3b_relu","res4a_branch1");
LG = connectLayers(LG,"bn4a_branch1","res4a/in2");
LG = connectLayers(LG,"bn4a_branch2b","res4a/in1");
LG = connectLayers(LG,"res4a_relu","res4b_branch2a");
LG = connectLayers(LG,"res4a_relu","res4b/in2");
LG = connectLayers(LG,"bn4b_branch2b","res4b/in1");
LG = connectLayers(LG,"res4b_relu","res5a_branch2a");
LG = connectLayers(LG,"res4b_relu","res5a_branch1");
LG = connectLayers(LG,"bn5a_branch1","res5a/in2");
LG = connectLayers(LG,"bn5a_branch2b","res5a/in1");
LG = connectLayers(LG,"res5a_relu","res5b_branch2a");
LG = connectLayers(LG,"res5a_relu","res5b/in2");
LG = connectLayers(LG,"bn5b_branch2b","res5b/in1");