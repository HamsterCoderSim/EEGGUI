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
        labelSubject
        labelTrail
        dataSubject
        dataTrail
    end
    
    methods
        function obj =  FeatureLoad(obj, showArea)
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
            obj.dataSubject = size(obj.classifyData, 1);
            obj.dataTrail = size(obj.classifyData, 2);
            showArea.SubjectNum.Text = num2str(size(obj.classifyData, 1)); 
            showArea.TrailNum.Text = num2str(size(obj.classifyData, 2));
            showArea.ChannelNum.Text = num2str(size(obj.classifyData, 3)); 
            showArea.Duration.Text = num2str(size(obj.classifyData, 4));
        end
        function obj = LabelLoad(obj, showArea)
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
            obj.labelSubject = size(obj.classifyLabel, 1);
            obj.labelTrail = size(obj.classifyLabel, 2);
            showArea.SubjectNum.Text = num2str(size(obj.classifyLabel, 1)); 
            showArea.TrailNum.Text = num2str(size(obj.classifyLabel, 2));
        end
        function obj = LeaveOneDataSplit(obj, methodSel, dataSel, leaveObj)
            %---先对数据进行调整，选择想要的数据---
            obj.classifyData = obj.classifyData(1:dataSel.Subject, 1:dataSel.Trail, :, :);
            obj.classifyLabel = obj.classifyLabel(1:dataSel.Subject, 1:dataSel.Trail);
            %------------传统的机器学习和迁移学习有一点区别------------
            trainTempData = []; trainTempLabel = []; testTempData = []; testTempLabel = [];
            [subjectNum, trailNum, ~, ~] = size(obj.classifyData);
            %数据分割主要因为向量需要输入的是向量，而其他的需要的是矩阵形式输入
            if methodSel == "SVM"
                for subject = 1:subjectNum
                    if subject == leaveObj
                        for trail = 1:trailNum
                            data = obj.classifyData(subject, trail, :, :);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                            label = obj.classifyLabel(subject, trail);
                            data = reshape(data, [1, size(data, 3)*size(data, 4)]);
                            label = ones(1, numel(data))*label;
                            testTempData = [testTempData, data]; testTempLabel = [testTempLabel, label];
                        end
                        obj.testData = testTempData;   obj.testLabel = testTempLabel;
                    else
                         for trail = 1:trailNum
                            data = obj.classifyData(subject, trail, :, :);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                            label = obj.classifyLabel(subject, trail);
                            data = reshape(data, [1, size(data, 3)*size(data, 4)]);
                            label = ones(1, numel(data))*label;
                            trainTempData = [trainTempData, data]; trainTempLabel = [trainTempLabel, label];
                        end
                       obj.trainData = trainTempData;   obj.trainLabel = trainTempLabel; 
                    end
                end
            else
                %---矩阵形式的数据--
                for subject = 1:subjectNum
                    if subject == leaveObj
                        for trail = 1:trailNum
                            data = obj.classifyData(subject, trail, :, :);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                            label = obj.classifyLabel(subject, trail);
                            data = reshape(data, [size(data, 3), size(data, 4)]);
                            label = ones(size(data, 1), 1)*label;
                            testTempData = [testTempData; data]; testTempLabel = [testTempLabel; label];
                        end
                        obj.testData = testTempData;   obj.testLabel = testTempLabel;
                    else
                        for trail = 1:trailNum
                            data = obj.classifyData(subject, trail, :, :);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                            label = obj.classifyLabel(subject, trail);
                            data = reshape(data, [size(data, 3), size(data, 4)]);
                            label = ones(size(data, 1), 1)*label;
                            trainTempData = [trainTempData; data]; trainTempLabel = [trainTempLabel; label];
                        end
                        obj.trainData = trainTempData;   obj.trainLabel = trainTempLabel;
                    end
                end
            end
        end
        function obj = NormalDataSplit(obj, splitVal, methodSel, dataSel)
            %---先对数据进行调整，选择想要的数据---
            obj.classifyData = obj.classifyData(1:dataSel.Subject, 1:dataSel.Trail, :, :);
            obj.classifyLabel = obj.classifyLabel(1:dataSel.Subject, 1:dataSel.Trail);
            %------------传统的机器学习和迁移学习有一点区别------------
            tempData = []; tempLabel = [];
            [subjectNum, trailNum, ~, ~] = size(obj.classifyData);
            %数据分割主要因为向量需要输入的是向量，而其他的需要的是矩阵形式输入
            if methodSel == "SVM"
                for subject = 1:subjectNum
                    for trail = 1:trailNum
                        data = obj.classifyData(subject, trail, :, 50);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                        label = obj.classifyLabel(subject, trail);
                        data = reshape(data, [1, size(data, 3)*size(data, 4)]);
                        label = ones(1, numel(data))*label;
                        tempData = [tempData, data]; tempLabel = [tempLabel, label];
                    end
                end
                randSort = randperm(numel(tempData));
                tempData = tempData(randSort); tempLabel = tempLabel(randSort);
                boundary = floor((splitVal/100)*numel(tempData));
                obj.trainData = tempData(1:boundary); obj.trainLabel = tempLabel(1:boundary);
                obj.testData = tempData(boundary+1:end); obj.testLabel = tempLabel(boundary+1:end);
            else
                %---矩阵形式的数据---
                for subject = 1:subjectNum
                    for trail = 1:trailNum
                        data = obj.classifyData(subject, trail, :, :);      %这个50是随便取的，因为训练太耗费时间了，所以就取一个数据
                        label = obj.classifyLabel(subject, trail);
                        data = reshape(data, [size(data, 3), size(data, 4)]);
                        label = ones(size(data, 1), 1)*label;
                        tempData = [tempData; data]; tempLabel = [tempLabel; label];
                    end
                end
                randSort = randperm(size(tempData, 1));
                tempData = tempData(randSort, :); tempLabel = tempLabel(randSort);
                boundary = floor((splitVal/100)*size(tempData, 1));
                obj.trainData = tempData(1:boundary, :); obj.trainLabel = tempLabel(1:boundary);
                obj.testData = tempData(boundary+1:end, :); obj.testLabel = tempLabel(boundary+1:end);
            end
        end
        function obj = MainClassify(obj, subjectSel, methodsSel, dataSel, textArea, lamp)     %最主要的部分
            lamp.Color = 'red'; pause(0.5);
            if strcmp(subjectSel.LeaveOne, 'On' )  %留一法
                subjectNum = size(obj.classifyData, 1);
                for leaveObj = 1:subjectNum
                    obj = obj.LeaveOneDataSplit(methodsSel, dataSel, leaveObj);
                    obj = obj.FeatureClassify(methodsSel);
                    obj.AccuracyDisp(textArea)
                end
            else     %普通分类方法
                obj = obj.NormalDataSplit(subjectSel.Normal, methodsSel, dataSel);
                obj = obj.FeatureClassify(methodsSel);
                obj.AccuracyDisp(textArea)
            end
        end
        function obj = FeatureClassify(obj, methodsSel)
            if methodsSel == "SVM";  obj = obj.SVMClassify; end
            if methodsSel == "RF"; obj = obj.RFClassify; end
            if methodsSel == "xgboost"; obj = obj.XGBClassify; end
            if methodsSel == "MEDA"; obj = obj.MedaClassify; end
            if methodsSel == "TCA"; obj = obj.TcaClassify; end
            if methodsSel == "SCA"; obj = obj.ScaClassify; end
            if methodsSel == "SA"; obj = obj.SaClassify; end
            if methodsSel == "TJM"; obj = obj.TjmClassify; end
            if methodsSel == "PCA"; obj = obj.PcaClassify; end
            if methodsSel == "MIDA"; obj = obj.MidaClassify; end
            if methodsSel == "LaplacianSVM"; obj = obj.LapSVMClassify; end
            if methodsSel == "LaplacianRidge"; obj = obj.LapRidgeClassify; end
            if methodsSel == "JDA"; obj = obj.JdaClassify; end
            if methodsSel == "ITL"; obj = obj.ItlClassify; end
            if methodsSel == "GFK"; obj = obj.GfkClassify; end
            if methodsSel == "EASYTL"; obj = obj.EasytlClassify; end
            if methodsSel == "CORALRF"; obj = obj.CoralRfClassify; end
            
        end
        function obj = AccuracyDisp(obj, textArea)
           textArea.Value = sprintf('the accuracy is %s\n', num2str(obj.classifyAccuracy)); 
        end
        function obj = SVMClassify(obj)
            % ---------设置参数，训练模型-----------
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
        end
        function obj = RFClassify(obj)
            %------------训练数据集---------------
            model = classRF_train(obj.trainData, obj.trainLabel, 1000, 2);
            yHat = classRF_predict(obj.testData, model);
            accuracy = length(find(yHat == obj.testLabel))/length(obj.testLabel);
            obj.classifyAccuracy = accuracy;
        end
        function obj = XGBClassify(obj)
            %----使用XGBoost进行分类-----
            params = struct;
            params.booster           = 'gbtree';
            params.objective         = 'binary:logistic';
            params.eta               = 0.1;
            params.min_child_weight  = 1;
            params.subsample         = 1; % 0.9
            params.colsample_bytree  = 1;
            params.num_parallel_tree = 1;
            params.max_depth         = 40;
            num_iters                = 500;
            model = xgboost_train(obj.trainData, obj.trainLabel, params, num_iters, 'None', []);
            yHat = xgboost_test(obj.testData, model,0);
            smallPosition = find(yHat <= 0.5); yHat(smallPosition) = 0;
            bigPosition = find(yHat > 0.5); yHat(bigPosition) = 1;
            accuracy = numel(find(yHat == obj.testLabel))/numel(obj.testLabel);
            obj.classifyAccuracy = accuracy;
        end
        function obj = MedaClassify(obj)
            columnNum = min(size(obj.trainData, 2), size(obj.testData, 2));obj.testData(:, 1:columnNum);
            onePosTrain = find(obj.trainLabel == 1); obj.trainLabel(onePosTrain) = 2;
            zeroPosTrain = find(obj.trainLabel == 0); obj.trainLabel(zeroPosTrain) = 1;
            onePosTest = find(obj.testLabel == 1); obj.testLabel(onePosTest) = 2;
            zeroPosTest = find(obj.testLabel == 0); obj.testLabel(zeroPosTest) = 1;
            %-----MEDA------
            options.d = 5;             %two important parameters to improve the classification, the first one
            options.rho = 1.0;
            options.p = 10;
            options.lambda = 10.0;
            options.eta = 0.05;         %the second one
            options.T = 10;
            [Acc,~,~,~] = MEDA(obj.trainData, obj.trainLabel, obj.testData, obj.testLabel, options);
            obj.classifyAccuracy = Acc;
        end
        function obj = TcaClassify(obj)
            %--------TCA--------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            param = []; param.kerName = 'lin'; param.bSstca = 0;
            param.mu = 1;param.m = 2;param.gamma = 0.1;param.lambda = 0;
            [Xproj, ~] = ftTrans_tca(X,maSrc', summaryLabel(maSrc), maSrc', param);            
            cvObj.training = maSrc'; cvObj.test = ~cvObj.training';
            acc = doPredict(Xproj(:,1:2), summaryLabel, cvObj);
            obj.classifyAccuracy = acc;
        end
        function obj = ScaClassify(obj)
            %-------------SCA----------
            [m, ~] = size(obj.trainData);
            boundary = 0.95*m;
            dataTrainCell{1} = obj.trainData(1:boundary, :);
            labelTrainCell{1} = obj.trainLabel(1:boundary);
            dataValidation = obj.trainData(boundary+1:end, :);
            labelValidation = obj.testLabel(boundary+1:end);           
            params.X_v = dataValidation;
            params.Y_v = labelValidation;
            params.verbose = true;
            [Acc] = SCA(dataTrainCell, labelTrainCell, dataTest, labelTest, params);
            obj.classifyAccuracy = Acc;
        end
        function obj = SaClassify(obj)
            %-------------------SA------------------------------------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(obj.trainData);
            boundary = 0.95*m;
            X = summaryData;%第一个参数
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc';cvObj.test = ~cvObj.training';
            param = []; param.pcaCoef = 2;
            [Xproj, ~] = ftTrans_sa(X,maSrc',summaryLabel(maSrc),maSrc',param);
            acc = doPredict(Xproj,summaryLabel,cvObj);
%             draw1(Xproj,summaryLabel,domainFt,{'z_1','z_2'},'SA',acc);
            obj.classifyAccuracy = acc;
        end
        function obj = TjmClassify(obj)
            %-----TJM------
            options.dim = 5;             %two important parameters to improve the classification, the first one
            options.kernel_type = 'linear';
            options.lambda = 15;
            options.T = 10;
            options.gamma = [];
            [Acc, ~, ~] = TJM(obj.trainData, obj.trainLabel, obj.testData, obj.testLabel, options);
            obj.classifyAccuracy = Acc;
        end
        function obj = PcaClassify(obj)
            %--------------PCA-----------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc';cvObj.test = ~cvObj.training';
            param = []; param.pcaCoef = 2; param.kerName = 'lin';
            [Xproj, ~] = ftTrans_pca(X, maSrc', summaryLabel(maSrc), maSrc', param);
            acc = doPredict(Xproj(:,1:2), summaryLabel, cvObj);
%             draw1(Xproj, summaryLabel, domainFt,{'z_1','z_2'}, 'PCA', acc)
            obj.classifyAccuracy = acc;
        end
        function obj = MidaClassify(obj)
            %-----------MIDA--------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc';cvObj.test = ~cvObj.training';
            param = []; param.kerName = 'lin';param.kerSigma = 1e-1;param.bSup = 0;
            param.mu = 1;param.m = 2;param.gamma = 1;
            [Xproj, ~] = ftTrans_mida(X,domainFt,summaryLabel(maSrc),maSrc',param);
            acc = doPredict(Xproj,summaryLabel,cvObj);
            obj.classifyAccuracy = acc;
%             draw1(Xproj,summaryLabel,domainFt,{'z_1','z_2'},'MIDA',acc)
        end
        function obj = LapSvmClassify(obj)
            %----------Laplacian SVM-------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc'; cvObj.test = ~cvObj.training';
            % Laplacian SVM
            param = []; param.t = 0;
            [pred, ~, ~] = mdlTrans_lapsvm(X(maSrc',:), summaryLabel(maSrc), X(~maSrc,:), param);
            acc = nnz(pred == summaryLabel(~maSrc))/length(pred);
            obj.classifyAccuracy = acc;
        end
        function obj = LapRidgeClassify(obj)
            %------------Laplacian ridge---------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc'; cvObj.test = ~cvObj.training';
            param = []; param.t = 0;
            [pred, ~] = mdlTrans_lapridge(X(maSrc',:),summaryLabel(maSrc), X(~maSrc',:),param);
            pred = (pred>1.5)+1;
            acc = nnz(pred == summaryLabel(~maSrc'))/length(pred);
            obj.classifyAccuracy = acc;
        end
        function obj = JdaClassify(obj)
            %-----JDA------
            options.dim = 5;             %two important parameters to improve the classification, the first one
            options.kernel_type = 'linear';
            options.lambda = 11;
            options.T = 10;
            options.gamma = [];
            [Acc, ~, ~] = JDA(obj.trainData, obj.trainLabel, obj.testData, obj.testLabel, options);
            obj.classifyAccuracy = Acc;
        end
        function obj = ItlClassify(obj)
            %------ITL-------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc';cvObj.test = ~cvObj.training';
            param = []; param.pcaCoef = 1; param.lambda = 10;
            [Xproj, ~] = ftTrans_itl(X,maSrc',summaryLabel(maSrc),maSrc',param);
            acc = doPredict(Xproj(:,1),summaryLabel,cvObj);
%             draw1([Xproj,Xproj*0],summaryLabel,domainFt,{'z_1',''},'ITL',acc)
            obj.classifyAccuracy = acc;
        end
        function obj = GfkClassify(obj)
            %--------GFK---------
            summaryData = [obj.trainData; obj.testData];
            summaryLabel = [obj.trainLabel; obj.testLabel];
            [m, ~] = size(summaryData);
            X = summaryData;%第一个参数
            boundary = size(obj.trainData, 1);
            maSrc(1:boundary) = true; maSrc(boundary+1:m) = false;  %第二个参数
            domainFt(:, 1) = maSrc';
            domainFt(:, 2) = ~(maSrc');
            cvObj.training = maSrc';cvObj.test = ~cvObj.training';
            % GFK
            param = []; param.dr = 1;
            [Xproj, ~] = ftTrans_gfk(X,maSrc',summaryLabel(maSrc),maSrc',param);
            acc = doPredict(Xproj(:,1:2),summaryLabel,cvObj);
%             draw1(Xproj,summaryLabel,domainFt,{'z_1','z_2'},'GFK',acc)
            obj.classifyAccuracy = acc;
        end
        function obj = EasytlClassify(obj)
            %-----EASYTL------
            % EasyTL with CORAL for intra-domain alignment
            [acc, ~] = EasyTL(dataTrain,labelTrain,dataTest,labelTest);
            obj.classifyAccuracy = acc;
        end
        function obj = CoralRfClassify(obj)
            %-----CORAL------
            Xs = double(obj.trainData);
            Xt = double(obj.testData);
            Ys = double(obj.trainLabel);
            Yt = double(obj.testLabel);
            cov_source = cov(Xs) + eye(size(Xs, 2));
            cov_target = cov(Xt) + eye(size(Xt, 2));
            A_coral = cov_source^(-1/2)*cov_target^(1/2);
            Sim_coral = double(Xs * A_coral * Xt');
            % RF
            %------------Start to train data with random forest---------------
            model = classRF_train(Xs, Ys, 1000, 2);
            yHat = classRF_predict(Xt, model);
            acc = length(find(yHat == Yt))/length(Yt);
            obj.classifyAccuracy = acc;
        end
    end
end

