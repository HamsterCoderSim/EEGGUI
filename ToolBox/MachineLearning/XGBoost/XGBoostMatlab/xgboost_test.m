function Yhat = xgboost_test(Xtest,model,loadmodel)
% eg:
% Yhat = xgboost_test(Xtest,model,1);
% figure; scatter(Yhat,ytest); grid on % make plot of labels vs predictions (probabilities of class==1 in case objective == 'binary:logistic'):
% 
% Train an xgboost model See "D:\cpcardio\physionet_2020\call_xgboost.m", "D:\r\xgboost\py\ex_xgboost.py"
% see also: xgboost_train.m

%%% Function inputs:
% model: a pointer. The output of xgboost_train.m
% Xtest: test dataset
% loadmodel : 0/1 - whether to load model from file or use the model in memory as create by xgboost_train.m

% NOTE: When loadmodel==0 (the model is read from memory), then pointer model.h_booster_ptr is NOT deleted after function ends. To delete the model
% from memory, use calllib('xgboost', 'XGBoosterFree',h_booster_ptr); clear h_booster_ptr . This is done by default for loadmodel==1.


if loadmodel == 0 % do not load model from file, but use pointer to the model in memory
    h_booster_ptr = model.h_booster_ptr; % the model already exists in memory, as created by xgboost_train.m
else % load model from file
    % load the xgboost library
    if not(libisloaded('xgboost'))
        cwd = pwd; cd D:\r\xgboost\lib
        loadlibrary('xgboost')
        cd(cwd)
    end
    % create the booster and set some parameters
    h_train_ptr = libpointer;
    h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);
    h_booster_ptr = libpointer;
    h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
    len = uint64(0); % set to == 0
    calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, len, h_booster_ptr_ptr);
    res = calllib('xgboost', 'XGBoosterLoadModel', h_booster_ptr_ptr, model.model_filename);
    if res == -1
        disp('Model could not be loaded')
        return
    end
end


%%%  Make predictions on a test set

rows = uint64(size(Xtest,1));
cols = uint64(size(Xtest,2));
Xtest = Xtest'; % DLL is row-based, and matlab is column-based

% set necesary pointers
h_test_ptr = libpointer;
h_test_ptr_ptr = libpointer('voidPtrPtr', h_test_ptr);
test_ptr = libpointer('singlePtr',single(Xtest));   

calllib('xgboost', 'XGDMatrixCreateFromMat', test_ptr, rows, cols, model.missing, h_test_ptr_ptr);

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

Yhat = double(f.Value); % display the predictions (in case objective == 'binary:logistic' : display the predicted probabilities)

if exist('h_test_ptr','var')
    calllib('xgboost', 'XGDMatrixFree',h_test_ptr); clear h_test_ptr
end

if loadmodel == 1
    if exist('h_booster_ptr','var')
        calllib('xgboost', 'XGBoosterFree',h_booster_ptr); clear h_booster_ptr
    end
end


