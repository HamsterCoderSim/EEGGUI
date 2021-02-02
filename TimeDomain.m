classdef TimeDomain
    %TimeDomain这个类是实现GUI界面中的数据载入与波形显示的类
    %   DataDisplay         是显示脑电数据的函数
    %   DataLoadFunction    是载入脑电数据的函数
    %   dataLoad            是每个被试者载入的数据格式要求为：subject x trail x channel x duration四维格式的数据
    %   channelNum          是脑电数据中的通道数
    %   trailNum            是做的实验的次数
    %   subjectNum          是被试者数目
    %   dataDispNum         是要显示的脑电数据，是一个向量：某次实验 x 某个通道 x duration
    %   durationNum         是脑电信号的长度
    properties
        dataLoad
        dataDispNum
        channelNum
        trailNum
        subjectNum
        durationNum
    end
    
    methods
        function [dataDisp] = DataDisplay(obj, showArea, memberSel, trailSel, channelSel)
            memberSel = regexp(memberSel, '\d', 'match'); memberSelArr = [];
            trailSel = regexp(trailSel, '\d', 'match');   trailSelArr = [];
            channelSel =regexp(channelSel, '\d', 'match'); channelSelArr = [];
            durationNumCell{1} = num2str(obj.durationNum);
            for k = 1:numel(memberSel); memberSelArr = [memberSelArr, memberSel{k}]; end
            for k = 1:numel(trailSel); trailSelArr = [trailSelArr, trailSel{k}]; end
            for k = 1:numel(channelSel); channelSelArr = [channelSelArr, channelSel{k}]; end
            if isempty(channelSel)
                channelSel = {1:obj.channelNum}; 
                dataDisp = obj.dataLoad(str2double(memberSelArr), str2double(trailSelArr), channelSel{1}, :);
            else
                dataDisp = obj.dataLoad(str2double(memberSelArr), str2double(trailSelArr), str2double(channelSelArr), :);
            end
            dataDisp = reshape(dataDisp, [size(dataDisp, 3), size(dataDisp, 4)]);
            if isempty(channelSelArr) 
                 showArea.subject.Text = memberSelArr; showArea.trail.Text = trailSelArr;
                 showArea.channel.Text = channelSelArr; showArea.duration.Text = durationNumCell;
                 showArea.lamp.Color = 'red';
            else
                showArea.subject.Text = memberSelArr; showArea.trail.Text = trailSelArr;
                showArea.channel.Text = channelSelArr; showArea.duration.Text = durationNumCell;
                showArea.lamp.Color = 'red';
            end
        end
        function obj = DataLoadFunction(obj)
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
                obj.subjectNum = 1;
                if numel(size(dataLoad)) > 2
                    obj.channelNum = size(dataLoad, 2);
                    obj.trailNum = size(dataLoad, 1);
                    obj.durationNum = size(dataLoad, 3);
                    obj.dataLoad = reshape(dataLoad, [1, size(dataLoad, 1), size(dataLoad, 2), size(dataLoad, 3)]);
                else
                    obj.channelNum = size(dataLoad, 1);
                    obj.durationNum = size(dataLoad, 2);
                    obj.trailNum = 1;
                    obj.dataLoad = reshape(dataLoad, [1, 1, size(dataLoad, 1), size(dataLoad, 2)]);
                end
            else
                h = waitbar(0, '开始导入', 'Name','进度条', 'WindowStyle', 'modal');
                for k=1:numel(FileName)
                    filePath = [PathName, FileName{k}];
                    data = load(filePath);
                    dataName = fieldnames(data);
                    data = data.('data');
                    if numel(size(data)) > 2
                        if k==1     %以第一次输入的长度为准
                            obj.durationNum = size(data, 3);
                            obj.channelNum = size(data, 2);
                            obj.trailNum = size(data, 1);
                        end
                        obj.dataLoad(k, : , :,1:obj.durationNum) = data;
                    else
                        if k==1     %以第一次输入的长度为准
                            obj.durationNum = size(data, 2);
                            obj.channelNum = size(data, 1);
                            obj.trailNum = 1;
                        end
                        obj.dataLoad(k, : , :, 1:obj.durationNum) = data;
                    end
                    waitbar(k/numel(FileName), h, ['已导入' num2str(k) '人']);
                end
                obj.subjectNum = numel(FileName);
            end
            close(h)
            delete(h);
            clear h;
        end
    end
end


