%{
    Created on 10:23 2020/1/20
    @author: XinWang Song
    This code is created to train the dataset with STL method.If you want
    to to learn more about this method, please visit this link:
    https://github.com/jindongwang/activityrecognition/tree/master/code/percom18_stl
%}
clear;
clc;
accuracySubject = zeros(1, 37);
for iteration=1:37
    trainDataTable = readtable('C:\Users\25626\Desktop\AMIGOS\DataTrain.xlsx', 'Sheet', iteration , 'ReadVariableNames',false);
    dataTrain = table2array(trainDataTable);
    trainLabelTable = readtable('C:\Users\25626\Desktop\AMIGOS\LabelTrain.xlsx','Sheet', iteration, 'ReadVariableNames',false);
    labelTrain = table2array(trainLabelTable);
    testDataTable = readtable('C:\Users\25626\Desktop\AMIGOS\DataTest.xlsx','Sheet', iteration, 'ReadVariableNames',false);
    dataTest = table2array(testDataTable);
    testLabelTable = readtable('C:\Users\25626\Desktop\AMIGOS\LabelTest.xlsx','Sheet', iteration, 'ReadVariableNames',false);
    labelTest = table2array(testLabelTable);
    onePosition = find(labelTrain == 1); labelTrain(onePosition) = 2;
    zeroPosition = find(labelTrain == 0); labelTrain(zeroPosition) = 1;
    onePosition = find(labelTest == 1); labelTest(onePosition) = 2;
    zeroPosition = find(labelTest == 0); labelTest(zeroPosition) = 1;
     %-----STL------
    dim = 5;
    [Acc] = STL(dataTrain, labelTrain, dataTest, labelTest, dim);
    fprintf('准确率%f\n', Acc);
    accuracySubject(iteration) = Acc;
end
bar(accuracySubject)
fprintf('平均准确率%f\n', mean(accuracySubject));