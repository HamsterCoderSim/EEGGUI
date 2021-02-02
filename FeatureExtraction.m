classdef FeatureExtraction
    %FeatureExtraction    是GUI界面中的特征提取模块
    %   featureData       是提取出的特征，数据格式为subject x trail x channel x duration
    %   eegDataNum        是用来接收TimeDomain这个类中传过来的数据，结构体中变量名字和TimeDomain中的一致
    %   FeatureExtraction 是构造函数，在这个构造函数中接收原始EEG
    %   DeExtraction      是提取脑电的微分熵特征函数
    
    properties
        featureData
        featureDataDisp
        eegDataNum
    end
    
    methods
        function obj = FeatureExtraction(timeDomObj, fsString, feaString)
             %--------------------提取合适的特征值------------------------
            eegDataNum = struct('dataLoad', timeDomObj.dataLoad, 'subjectNum',...
                timeDomObj.subjectNum, 'trailNum', timeDomObj.trailNum, 'channelNum',...
                timeDomObj.channelNum, 'fs', str2double(fsString));
            if feaString == "微分熵"; obj = obj.DeExtraction(eegDataNum); end
        end
        function obj = DeExtraction(obj, eegDataNum)
            windowLength = size(eegDataNum.dataLoad, 2)/2;
            Parameters.data = eegDataNum.dataLoad;
            h = waitbar(0, '特征提取', 'Name','进度条', 'WindowStyle', 'modal');
            for subject = 1:eegDataNum.subjectNum
                waitbar(subject/(eegDataNum.subjectNum), h, ['已提取' num2str(subject*100/(eegDataNum.subjectNum)) '%']);
                for trail = 1:eegDataNum.trailNum                   
                    Parameters.Fs = eegDataNum.fs; Parameters.nOverlap =0; 
                    Parameters.Nfft = eegDataNum.fs;     %滤波后还用128？
                    Parameters.window = hanning(windowLength)';
                    dataTemp = Parameters.data(subject, trail, :, :);
                    dataTemp = reshape(dataTemp, [size(dataTemp, 3), size(dataTemp, 4)]);
                    dataTemp = dataTemp';
                    dataTempLen = size(dataTemp, 1)/Parameters.Fs; 
                    deMatrix = [];
                    for length = 1:dataTempLen-1
                        dataTempDiv = dataTemp(length*(Parameters.Fs):(length+1)*Parameters.Fs, :);     %每秒钟计算一个特征值，最后的一部分可以直接省掉，因为最后一秒的影响并不大
                        [Pxx, f] = pwelch(dataTempDiv, Parameters.window, Parameters.nOverlap, Parameters.Nfft, Parameters.Fs);
                        de = 0.5*log2(sum(Pxx, 1));
                        de = (mapminmax(de));     %归一化处理
                        deMatrix = [deMatrix; de];
                    end
                    featureDataVal(subject, trail, 1:size(deMatrix', 1), 1:size(deMatrix', 2)) = deMatrix';
                end
            end
            obj.featureData = featureDataVal;
            close(h)
            delete(h);
            clear h;
        end
    end
end

