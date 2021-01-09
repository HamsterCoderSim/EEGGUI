classdef TimeDomain
    %TimeDomain这个类是实现GUI界面中的数据载入与波形显示的类
    %   DataDisplay         是显示脑电数据的函数
    %   DataLoadFunction    是载入脑电数据的函数
    %   dataLoadNum         是每个被试者载入的数据格式要求为：subject x trail x channel x trail三维格式的数据
    %   channelNum          是脑电数据中的通道数
    %   trailNum            是做的实验的次数
    %   subjectNum          是被试者数目
    %   dataDispNum         是要显示的脑电数据，是一个向量：某次实验 x 某个通道 x duration
    properties
        dataLoadNum
        dataDispNum
        channelNum
        trailNum
        subjectNum
    end
    
    methods
        function [dataDisp] = DataDisplay(obj, textArea, memberSel, trailSel, channelSel)
            memberSel = regexp(memberSel, '\d', 'match'); memberSelArr = [];
            trailSel = regexp(trailSel, '\d', 'match');   trailSelArr = [];
            channelSel =regexp(channelSel, '\d', 'match'); channelSelArr = [];
            for k = 1:numel(memberSel); memberSelArr = [memberSelArr, memberSel{k}]; end
            for k = 1:numel(trailSel); trailSelArr = [trailSelArr, trailSel{k}]; end
            for k = 1:numel(channelSel); channelSelArr = [channelSelArr, channelSel{k}]; end
            if isempty(channelSel)
                channelSel = {1:obj.channelNum}; 
                dataDisp = obj.dataLoadNum(str2double(memberSelArr), str2double(trailSelArr), channelSel{1}, :);
            else
                dataDisp = obj.dataLoadNum(str2double(memberSelArr), str2double(trailSelArr), str2double(channelSelArr), :);
            end
            dataDisp = reshape(dataDisp, [size(dataDisp, 3), size(dataDisp, 4)]);
            if isempty(channelSelArr)
                 textArea.Value = ['当前选择第', memberSelArr, '位被试', '第', trailSelArr, '次实验',  '所有通道的数据'];
            else
                textArea.Value = ['当前选择第', memberSelArr, '位被试', '第', trailSelArr, '次实验', '第', channelSelArr, '个通道的数据']; 
            end
        end
        function [dataLoad, subjectNum, trailNum, channelNum] = DataLoadFunction(obj)
            [FileName, PathName] = uigetfile('.mat', 'MultiSelect', 'on');
            if isempty(FileName)
                errordlg('请选择文件', '错误提示');
            end
            if ~iscell(FileName)
                filePath = [PathName, FileName];
                if FileName==0 ; errordlg('请选择数据', '错误提示'); end
                data = load(filePath);
                dataName = fieldnames(data);
                dataLoad = data.('data');
                subjectNum = 1;
                channelNum = size(dataLoad, 2);
                trailNum = size(dataLoad, 1);
                dataLoad = reshape(dataLoad, [1, size(dataLoad, 1), size(dataLoad, 2), size(dataLoad, 3)]);     
            else
                for k=1:numel(FileName)
                    filePath = [PathName, FileName{k}];
                    data = load(filePath);
                    dataName = fieldnames(data);
                    data = data.('data');
                    if k==1     %以第一次输入的长度为准
                        signalLen = size(data, 3);
                        channelNum = size(data, 2);
                        trailNum = size(data, 1);
                    end
                    dataLoad(k, : , :,1:signalLen) = data;
                end
                subjectNum = numel(FileName);
            end   
        end
    end
end


