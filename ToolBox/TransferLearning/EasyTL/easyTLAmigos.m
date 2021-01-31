%{
    Created on 19:25 2020/1/20
    @author: XinWang Song
    This code is created to train the dataset with EASYTL method.If you want
    to to learn more about this method, please visit this link:
    https://github.com/jindongwang/transferlearning/tree/master/code/traditional/EasyTL
%}
clear;
clc;
list_acc = [];
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
    %-----EASYTL------
    
    % EasyTL without intra-domain alignment [EasyTL(c)]
    [Acc1, ~] = EasyTL(dataTrain,labelTrain,dataTest,labelTest,'raw');
    fprintf('Acc: %f\n',Acc1);
    disp('------');
    % EasyTL with CORAL for intra-domain alignment
    [Acc2, ~] = EasyTL(dataTrain,labelTrain,dataTest,labelTest);
    fprintf('Acc: %f\n',Acc2);
    list_acc = [list_acc;[Acc1,Acc2]];
    
    fprintf('the accuracy is %f\n', Acc2);
    accuracySubject(iteration) = Acc2;
end
bar(accuracySubject)
fprintf('平均准确率%f\n', mean(accuracySubject));