%Michael Ryan 
%CPE 440 Final Project 
%First block of code will set up our data
%path will be used to pull the project data
path = 'C:\Users\Test\Documents\AI Final\Project Data';
%We will use imageDatastore to convert the raw project data into format
%matlab can use
data = imageDatastore(path, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%split data into testing and training sets
[trainData,testData] = splitEachLabel(data,0.3,'randomize');

%%
%CNN Classifier
%We will use alexnet to train our own neural network using transfer
%learning
net = alexnet;
options = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 15, 'MiniBatchSize', 32, ... 
    'Plots', 'training-progress');
%alter alexnet slightly to accomodate our classification of three classes
%only
layers = net.Layers;
layers(23) = fullyConnectedLayer(3);
layers(25) = classificationLayer;
%resize both sets of data according to the imageSize required by alexnet
imageSize = net.Layers(1).InputSize;
resizedTrainingData = augmentedImageDatastore(imageSize, trainData);
resizedTestData = augmentedImageDatastore(imageSize, testData);
%train new neural network through transfer learning
newNet = trainNetwork(resizedTrainingData, layers, options);
%classify using new CNN by inputting testData
cnnPredictedLabels = classify(newNet, resizedTestData);
%output statistics
cnnAccuracy = mean(cnnPredictedLabels == testData.Labels);
confMat = confusionmat(testData.Labels, cnnPredictedLabels);
%%
%SVM Classifier
%Rather than train a new SVM classifier from scratch which would take a lot of resources, we will use the
%feautres extracted by our CNN which we trained above.
%We can choose whichever layer we want, but for best accuracy, the layer
%directly before classification would have extracted the "best" features
featureLayer = 'prob';
%pull features from featureLayer
trainingFeatures = activations(newNet, resizedTrainingData, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%train multi-class svm using extracted trainingFeatures
classifier = fitcecoc(trainingFeatures, trainData.Labels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%pull features from testing data
testFeatures = activations(newNet, resizedTestData, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%classify using the testFeatures
svmPredictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
%output statistics
svmConfMat = confusionmat(testData.Labels, svmPredictedLabels);
svmAccuracy = mean(svmPredictedLabels == testData.Labels);
%It is also possible to train an SVM model from scratch using matlab's
%built in feature extraction. This can be more accurate, but is also
%significantly more computationally expensive.

