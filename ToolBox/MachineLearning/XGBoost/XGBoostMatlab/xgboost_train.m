function model = xgboost_train(Xtrain,ytrain,params,max_num_iters,eval_metric,model_filename)
%%% eg:
% load carsmall; Xtrain = [Acceleration Cylinders Displacement Horsepower MPG]; ytrain = cellstr(Origin); ytrain = double(ismember(ytrain,'USA'));
% X = Xtrain(1:70,:); y = ytrain(1:70); Xtest = Xtrain(size(X,1)+1:end,:); ytest = ytrain(size(X,1)+1:end);
% model_filename = []; model = xgboost_train(X,y,[],999,'AUC',model_filename); %%% model_filename = 'xgboost_model.xgb'
% loadmodel = 0; Yhat = xgboost_test(Xtest,model,loadmodel);
% [XX,YY,~,AUC] = perfcurve(ytest,Yhat,1);
% figure; plot(XX,YY,'LineWidth',2); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Classification by Logistic Regression'); grid on
% figure; scatter(Yhat,ytest + 0.1*rand(length(ytest),1)); grid on
% unloadlibrary('xgboost')
%
% eg:
skipme = 1;
if skipme == 0
    load carsmall; Xtrain = [Acceleration Cylinders Displacement Horsepower MPG]; ytrain = cellstr(Origin); ytrain = double(ismember(ytrain,'USA'));
    Xtrain = [25,93,43;5,90,98;66,73,49]; ytrain = [0,1,1]';
    xTest = [69,36,40;51,5,99]; yTest = [0,0]';
    params = struct;
    params.booster           = 'gbtree';
    params.objective         = 'binary:logistic';
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 1; % 0.9
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
    params.max_depth         = 5;
    num_iters                = 3;
    model = xgboost_train(Xtrain,ytrain,params,num_iters,'None',[]);
    yHat = xgboost_test(xTest,model,0);
    accuracy = numel(find(yHat == yTest))/numel(yTest);
    fprintf('正确率是%f\n', accuracy);
    figure; plot(yHat)
    figure; scatter(yHat,ytrain + 0.1*rand(length(ytrain),1)); grid on
end


%%% Function inputs:
% Xtrain:        matrix of inputs for the training set
% ytrain:        vetor of labels/values for the test set
% params:        structure of learning parameters
% max_num_iters: max number of iterations for learning
% eval_metric:   evaluation metric for cross-validation performance regarding turning the optimal number of learning iterations. 
                 % Suppoted are 'AUC', 'Accuracy', 'None'.
                 % In case eval_metric=='None', learning of final model will be performed with max_num_iters (without internal cross validation).
                 % For other evaluation metrics, up to max_num_iters learning iterations will be performed in a cross-validation procedure.
                 % In each cross-validation fold, learning will the stopped if eval_metric is not improved over last early_stopping_rounds (number of) iterations.
                 % Then, learning of the final model will be performed using the average (over CV folds) number of resulting learning iterations.
% model_filename : a string. If nonempty or ~= '', the final name will be saved to a file specified by this string 
%%% Function output:
% model: a structure containing:
%     iters_optimal; % number of iterations performs by xgboost (final model)
%     h_booster_ptr; % pointer to the final model
%     params;        % model parameters (just for info)
%     missing;       % value considered "missing"

% train an xgboost model See "D:\cpcardio\physionet_2020\call_xgboost.m", "D:\r\xgboost\py\ex_xgboost.py"
% see also: xgboost_test.m

% See https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html for info on the xgboost library functions
% See https://xgboost.readthedocs.io/en/latest/parameter.html     for info on xgboost inputs parameters

%%% Steps to compile xgboost library and use it in Matlab:
    % Step 1: create xgboost.dll (on windows)
    % Follow these instructions: https://xgboost.readthedocs.io/en/latest/build.html#build-the-shared-library
    %   - make folder D:\r\xgboost (e.g.)
    %   - create an empty git repository 
    %   - pull from https://github.com/dmlc/xgboost
    % 	- Git bash here (D:\r\xgboost) - open a git bash. In it type:
    % 	- git submodule init
    % 	- git submodule update
    % 	- install cmake and add path to the env (automatically, just select the option)
    %        = https://cgold.readthedocs.io/en/latest/first-step/installation.html
    %        = download and install: https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-win64-x64.msi
    %   - In dos, go to folder D:\r\xgboost. In it execute:
    % 		mkdir build
    % 		cd build
    % 		cmake .. -G"Visual Studio 14 2015 Win64"
    % 		# for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
    % 		# for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
    % 		cmake --build . --config Release
    % 
    %       Result:
    %       xgboost.dll is created here: "D:\r\xgboost\lib\xgboost.dll"
    %
    % Step 2: get a header file:
    %   - downlaod header file: https://raw.githubusercontent.com/dmlc/xgboost/master/include/xgboost/c_api.h
    %   - save it to "D:\r\xgboost\lib"
    %   - rename c_api.h to xgboost.h
    %
    %       Result:
    %       xgboost.h is created here: "D:\r\xgboost\lib\xgboost.h"
    %
    % Step 3: run this file (xgboost_train.m) to produce a model and xgboost_test.m to make predictions using the model created in xgboost_train.m
    %   - The script utilizes an explanation on "Using XGBOOST in c++" provided here: https://stackoverflow.com/questions/36071672/using-xgboost-in-c


% Example code to perform cross-validation for tuning (some) parameters:
skipme = 1;
if skipme == 0
    %%% load example data:
    % load carsmall; Xtrain = [Acceleration Cylinders Displacement Horsepower MPG]; ytrain = cellstr(Origin); ytrain = double(ismember(ytrain,'USA'));
    
    save_model_to_disk = 0; % 0 or 1 - whether to save model to disk (==1) or not (==0)
    eval_metric        = 'Accuracy'; % Out-of-sample evaluation metric. One of: 'Accuracy', 'AUC'
    
    if save_model_to_disk == 1
        model_filename = 'xgb_model.xgb';
    else
        model_filename = [];
    end
    
    % create CV fold indices
    folds = 5;
    cvind = ceil(folds*[1:size(Xtrain,1)]/(size(Xtrain,1)))'; % column containing the folder indices for cross validation
    rand('state', 0); u1 = rand(size(Xtrain,1),1); cvind = sortrows([u1 , cvind],1); cvind = cvind(:,2:end); clear u1
    
    % create params structure
    params = struct;
    params.booster           = 'gbtree';
    params.objective         = 'binary:logistic';
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 1; % 0.9
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
    
    % set range of possible values for (some) entries of the params structure
	params.max_depth_all     = [1:5];
    num_iters_all            = 2.^(1:10); % n_estimators
    
    % perform search over the range of parameters
    CVACC = []; % cross validation accuracy criterion
    CVAUC = []; % cross validation AUC criterion
    for i=1:length(params.max_depth_all)
        params.max_depth = params.max_depth_all(i);
        for j=1:length(num_iters_all)
            disp(['i=' num2str(i) '/' num2str(length(params.max_depth_all)) ', j=' num2str(j) '/' num2str(length(num_iters_all))])
            YhatCV_all = zeros(size(Xtrain,1),1);
            for kk=1:folds % perform a cross-validation step for params.max_depth_all(i) and num_iters_all(j)
                num_iters = num_iters_all(j);
                model = xgboost_train(Xtrain(cvind~=kk,:),ytrain(cvind~=kk),params,num_iters,'None',[]);
                YhatCV_all(cvind==kk) = xgboost_test(Xtrain(cvind==kk,:),model,0);
            end
            if strcmp(eval_metric,'Accuracy')
                CVACC = [CVACC; [sum(ytrain == round(YhatCV_all))/length(ytrain) params.max_depth num_iters]]; % cross-validation accuracy
            elseif strcmp(eval_metric,'AUC')
                [~,~,~,AUC] = perfcurve(ytrain,YhatCV_all,0);
                CVAUC = [CVAUC; [AUC params.max_depth num_iters]];
            end
        end
    end
    if strcmp(eval_metric,'Accuracy')
        CV_metric_optimal = CVACC(CVACC(:,1) == max(CVACC(:,1)),:); % [CV_accuracy max_depth num_iters]
    elseif strcmp(eval_metric,'AUC')
        CV_metric_optimal = CVAUC(CVAUC(:,1) == max(CVAUC(:,1)),:); % [CV_AUC max_depth num_iters]
    end
    params.max_depth = CV_metric_optimal(1,2);
    num_iters        = CV_metric_optimal(1,3);
    % train the whole model using optimal parameters
    model = xgboost_train(Xtrain,ytrain,params,num_iters,'None',model_filename);
    YhatTrain = xgboost_test(Xtrain,model,save_model_to_disk); % test on the training set (just for info)
    if isfield(model,'h_booster_ptr')
        calllib('xgboost', 'XGBoosterFree',model.h_booster_ptr); % clear model data from memory. Do this only ONCE, otherwise Matlab will hang.
        model = rmfield(model,'h_booster_ptr');
    end
    % plot original labels vs predicted probabilities
    figure; scatter(YhatTrain, ytrain + 0.1*rand(length(ytrain),1)); grid on
    % Plot CV results:
    if strcmp(eval_metric,'Accuracy')
        CVperf = CVACC;
        CV_performance = 'CV Accuracy';
    elseif strcmp(eval_metric,'AUC')
        CVperf = CVAUC;
        CV_performance = 'CV AUC';
    end
    [Z1,Z2] = meshgrid(params.max_depth_all,num_iters_all);
    Z = reshape(CVperf(:,1),size(Z1,1),size(Z1,2));
    figure; surfc(Z1,Z2,Z); alpha(0.2); xlabel('max depth'); ylabel('num iters'), zlabel(CV_performance)
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% set some parameters manually:
early_stopping_rounds = 10; % use CV with early_stopping_rounds== [10]
folds = 5;                  % number of cross validation folds
missing = single(NaN);      % set a value to be treated as "missing" 

% load example data, if not supplied:
if isempty(Xtrain)
    load carsmall
    Xtrain = [Acceleration Cylinders Displacement Horsepower MPG]; % contains input data WITHOUT labels
    ytrain = cellstr(Origin); % set the labels (target variable)
    ytrain = double(ismember(ytrain,'USA'));
end

% set max number of iterations for learning
if isempty(max_num_iters)
    max_num_iters = 999; % == num_boost_round
end    
        
% parse xgboost parameters:
%%% default in https://stackoverflow.com/questions/36071672/using-xgboost-in-c:
if isempty(params)
    params.booster           = 'gbtree';
    params.objective         = 'binary:logistic';
    params.max_depth         = 5;
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 0.9;
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
end
%%% default in "D:\g\py\hot\postprob\postprob.py", "D:\r\xgboost\py\ex_xgboost.py":
% if isempty(params)
%     % params.n_estimators      = num2str(1000); % option only available in python
%     % params.nthred            = num2str(3); % option only available in python
%     params.booster           = 'gbtree';
%     params.objective         = 'binary:logistic';
%     params.scale_pos_weight  = num2str(1);
%     params.subsample         = num2str(0.9);
%     params.gamma             = num2str(0);
%     params.reg_alpha         = num2str(0);
%     params.max_depth         = num2str(5);
% end

param_fields = fields(params);
for i=1:length(param_fields)
    eval(['params.' param_fields{i} ' = num2str(params.' param_fields{i} ');'])
end

% load the xgboost library
if not(libisloaded('xgboost'))
    cwd = pwd; cd F:\MATLAB2020b\MachineLearning\xgboost\lib
    loadlibrary('xgboost.dll', 'xgboost.h')
    cd(cwd)
end

if ~strcmp(eval_metric,'None')
    cvind = ceil(folds*[1:size(Xtrain,1)]/(size(Xtrain,1)))'; % column containing the folder indices for cross validation
    rand('state', 0); u1 = rand(size(Xtrain,1),1); cvind = sortrows([u1 , cvind],1); cvind = cvind(:,2:end); clear u1
    iters_reached_per_fold = zeros(folds,1);
    for kk = 1:folds

        % post-process input data
        rows = uint64(sum(cvind~=kk)); % use uint64(size(Xtrain,1)) in case of no CV
        cols = uint64(size(Xtrain,2));

        % create relevant pointers
        train_ptr = libpointer('singlePtr',single(Xtrain(cvind~=kk,:)')); % the transposed (cv)training set is supplied to the pointer
        train_labels_ptr = libpointer('singlePtr',single(ytrain(cvind~=kk)));

        h_train_ptr = libpointer;
        h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);

        % convert input matrix to DMatrix
        calllib('xgboost', 'XGDMatrixCreateFromMat', train_ptr, rows, cols, missing, h_train_ptr_ptr);

        % handle the labels (target variable)
        labelStr = 'label';
        calllib('xgboost', 'XGDMatrixSetFloatInfo', h_train_ptr, labelStr, train_labels_ptr, rows);

        % create the booster and set some parameters
        h_booster_ptr = libpointer;
        h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
        len = uint64(1);

        calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, len, h_booster_ptr_ptr);
        for i=1:length(param_fields)
            eval(['calllib(''xgboost'', ''XGBoosterSetParam'', h_booster_ptr, ''' param_fields{i} ''', ''' eval(['params.' param_fields{i}]) ''');'])
        end
        %%% for example:
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'booster', 'gbtree');
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'objective', 'binary:logistic'); % 'reg:linear' , 'binary:logistic'
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'max_depth', '5');
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'eta', '0.1');
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'min_child_weight', '1');
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'subsample', '1'); % '1', '0.5'
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'colsample_bytree', '1');
        % calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'num_parallel_tree', '1');

        %%%  Make a (cv) model

        % see https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
        % see https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
        %%% calllib('xgboost', 'XGBoosterSetParam', h_booster_ptr, 'eval_metric', 'logloss'); % eg: 'logloss' , 'auc' , 'mae'. NOTE: there is no way to provide early_stopping_rounds inside params

        % initialize
        AUC_ = []; % AUC      - for CV performance evaluation
        Acc_ = []; % Accuracy - for CV performance evaluation

        % create a test set
        h_test_ptr = libpointer;
        h_test_ptr_ptr = libpointer('voidPtrPtr', h_test_ptr);
        test_ptr = libpointer('singlePtr',single(Xtrain(cvind==kk,:)')); % the transposed (cv)training set is supplied to the pointer
        yCV      = ytrain(cvind==kk); % not supplied to xgboost.dll
        rows = uint64(sum(cvind==kk)); % use uint64(size(Xtrain,1)) in case of no CV
        cols = uint64(size(Xtrain,2));
        calllib('xgboost', 'XGDMatrixCreateFromMat', test_ptr, rows, cols, missing, h_test_ptr_ptr);

        % perform up to max_num_iters learning iterations. Stop learning if eval_metric is not improved over last early_stopping_rounds (number of) iterations
        for iter = 0:max_num_iters
            % disp(['iter (cv ' num2str(kk) ') = ' num2str(iter)])
            calllib('xgboost', 'XGBoosterUpdateOneIter', h_booster_ptr, int32(iter), h_train_ptr);

            %%%  Make predictions on a CV test set

            % predict
            out_len = uint64(0);
            out_len_ptr = libpointer('uint64Ptr', out_len);
            f = libpointer('singlePtr');
            f_ptr = libpointer('singlePtrPtr', f);
            option_mask = int32(0);
            ntree_limit = uint32(0);
            training = int32(0);
            calllib('xgboost', 'XGBoosterPredict', h_booster_ptr, h_test_ptr, option_mask, ntree_limit, training, out_len_ptr, f_ptr);

            % extract predictions
            n_outputs = out_len_ptr.Value;
            setdatatype(f,'singlePtr',n_outputs);

            YhatCV = double(f.Value); % display the predictions (in case objective == 'binary:logistic' : display the predicted probabilities)
            % YhatCV = round(YhatCV); % so that we get the label

            switch eval_metric
                case 'AUC'
                    % use AUC as evaluation metric
                    [~,~,~,AUC] = perfcurve(yCV,YhatCV,1);
                    AUC_ = [AUC_; AUC];
                    if length(AUC_) > early_stopping_rounds && AUC_(iter-early_stopping_rounds+2) == max(AUC_(iter-early_stopping_rounds+2:end))
                        iters_reached_per_fold(kk) = iter-early_stopping_rounds+2;
                        break
                    end
                case 'Accuracy'
                    % use Accuracy as evaluation metric
                    Acc = [sum(yCV == round(YhatCV)) / length(yCV)];
                    Acc_ = [Acc_; Acc];
                    if length(Acc_) > early_stopping_rounds && Acc_(iter-early_stopping_rounds+2) == max(Acc_(iter-early_stopping_rounds+2:end))
                        iters_reached_per_fold(kk) = iter-early_stopping_rounds+2;
                        break
                    end
                otherwise
                    % free xgboost internal structures
                    if exist('h_train_ptr','var')
                        calllib('xgboost', 'XGDMatrixFree',h_train_ptr); clear h_train_ptr
                    end
                    if exist('h_test_ptr','var')
                        calllib('xgboost', 'XGDMatrixFree',h_test_ptr); clear h_test_ptr
                    end
                    if exist('h_booster_ptr','var')
                        calllib('xgboost', 'XGBoosterFree',h_booster_ptr); clear h_booster_ptr
                    end
                    disp('Evaluation metric not supported')
                    return
            end
        end

        % free xgboost internal structures
        if exist('h_train_ptr','var')
            calllib('xgboost', 'XGDMatrixFree',h_train_ptr); clear h_train_ptr
        end
        if exist('h_test_ptr','var')
            calllib('xgboost', 'XGDMatrixFree',h_test_ptr); clear h_test_ptr
        end
        if exist('h_booster_ptr','var')
            calllib('xgboost', 'XGBoosterFree',h_booster_ptr); clear h_booster_ptr
        end
    end
    iters_optimal = round(mean(iters_reached_per_fold)); % estimated optimal number of learning iterations for the whole training set
    disp('optimal iterations per cv fold:')
    disp(iters_reached_per_fold)
else
    iters_optimal = max_num_iters;
end


%%% Train the final model using the whole training set and number of iterations == iters_optimal:

% post-process input data
rows = uint64(size(Xtrain,1)); % use uint64(size(Xtrain,1)) in case of no CV
cols = uint64(size(Xtrain,2));
Xtrain = Xtrain'; % DLL is row-based, and matlab is column-based

% create relevant pointers
train_ptr = libpointer('singlePtr',single(Xtrain));
train_labels_ptr = libpointer('singlePtr',single(ytrain));

h_train_ptr = libpointer;
h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);

% convert input matrix to DMatrix
calllib('xgboost', 'XGDMatrixCreateFromMat', train_ptr, rows, cols, missing, h_train_ptr_ptr);

% handle the labels (target variable)
labelStr = 'label';
calllib('xgboost', 'XGDMatrixSetFloatInfo', h_train_ptr, labelStr, train_labels_ptr, rows);

% create the booster and set some parameters
h_booster_ptr = libpointer;
h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
len = uint64(1);

calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, len, h_booster_ptr_ptr);
for i=1:length(param_fields)
    eval(['calllib(''xgboost'', ''XGBoosterSetParam'', h_booster_ptr, ''' param_fields{i} ''', ''' eval(['params.' param_fields{i}]) ''');'])
end

% perform iters_optimal learning iterations to produce a final model (pointer to it)
for iter = 0:iters_optimal
    % disp(['iter (final model) = ' num2str(iter)])
    calllib('xgboost', 'XGBoosterUpdateOneIter', h_booster_ptr, int32(iter), h_train_ptr);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model                = struct;
model.iters_optimal  = iters_optimal; % number of iterations performs by xgboost
model.h_booster_ptr  = h_booster_ptr; % pointer to the final model
model.params         = params;        % just for info
model.missing        = missing;       % value considered "missing"
model.model_filename = '';            % initialize: filename for model (to be saved)

if ~(isempty(model_filename) || strcmp(model_filename,''))
    calllib('xgboost', 'XGBoosterSaveModel', h_booster_ptr_ptr, model_filename);
    model.model_filename = model_filename; % 'xgboost_model.xgb'
end

% free xgboost internal structures
if exist('h_train_ptr','var')
    calllib('xgboost', 'XGDMatrixFree',h_train_ptr); clear h_train_ptr
end
if exist('h_booster_ptr','var')
    % calllib('xgboost', 'XGBoosterFree',h_booster_ptr); clear h_booster_ptr
end

% % unload the xgboost library
% if libisloaded('xgboost')
%     unloadlibrary('xgboost')
% end





