classdef EEGClassify
    %UNTITLED 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        classifyData
        classifyLabel
        classifyAccuracy
        trainData
        trainLabel
        testData
        testLabel
    end
    
    methods
        function obj =  FeatureLoad(obj)
            [FileName, PathName] = uigetfile('.mat', 'MultiSelect','on');
            data = load([PathName, FileName]);
            h = waitbar(0, '特征提取', 'Name','进度条', 'WindowStyle', 'modal');
            varName = fieldnames(data);
            obj.classifyData= data.(varName{1});
            waitbar(100, h, ['读取完成' num2str(100) '%']);
            pause(2);
            close(h)
            delete(h);
            clear h;
        end
        function obj = LabelLoad(obj)
            [FileName, PathName] = uigetfile('.mat', 'MultiSelect','on');
            h = waitbar(0, '特征提取', 'Name','进度条', 'WindowStyle', 'modal');
            label = load([PathName, FileName]);
            varName = fieldnames(label);
            obj.classifyLabel = label.(varName{1});
            waitbar(100, h, ['读取完成' num2str(100) '%']);
            pause(2);
            close(h)
            delete(h);
            clear h;
        end
        function obj = DataSplit(obj, splitVal, methodSel)
            %------------传统的机器学习和迁移学习有一点区别------------
            tempData = []; tempLabel = [];
            
            %数据分割主要因为向量需要输入的是向量，而其他的需要的是矩阵形式输入
            if methodSel == "SVM"
                [subjectNum, trailNum, ~, ~] = size(obj.classifyData);
                for subject = 1:subjectNum
                    for trail = 1:trailNum
                        data = obj.classifyData(subject, trail, :, 50);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                        label = obj.classifyLabel(subject, trail);
                        data = reshape(data, [size(data, 4), size(data, 3)]);
                        label = ones(1, numel(data))*label;
                        tempData = [tempData, data]; tempLabel = [tempLabel, label];
                    end
                end
                boundary = floor((splitVal/100)*numel(tempData));
                obj.trainData = tempData(1:boundary); obj.trainLabel = tempLabel(1:boundary);
                obj.testData = tempData(boundary+1:end); obj.testLabel = tempLabel(boundary+1:end);
            else            
            end
            %----------迁移学习主要是通过被试者个数来分割的，所以这里从这分开
        end
        function obj = FeatureClassify(obj, methodsSel)
            if methodsSel == "SVM";  obj = obj.SVMClassify; end
        end
        function obj = SVMClassify(obj)
            % ---------设置参数，训练模型-----------
            dfile = '日志文件.txt';
            if exist(dfile, 'file'); delete dfile; end
            diary(dfile)
            diary on
            C = [0.001 0.01 0.1 1.0 10 100 ];
            parfor i = 1 :size(C,2)
                svmModel(i) = libsvmtrain(double((obj.trainLabel)'), sparse(double((obj.trainData)')),sprintf('-c %d -q -v 2',C(i) ));
            end
            [~, indx]=max(svmModel);
            CVal = C(indx);
            svmModel = libsvmtrain(double((obj.trainLabel)'), sparse(double((obj.trainData)')),sprintf('-c %d -q',CVal));
            %-----------使用训练好的模型分类------------
            [~, accuracy, ~] = libsvmpredict((obj.testLabel)', (obj.testData)', svmModel);
            obj.classifyAccuracy = accuracy(1,1);
            diary off
        end
    end
end

