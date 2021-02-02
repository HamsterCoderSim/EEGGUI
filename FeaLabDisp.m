classdef FeaLabDisp
    %UNTITLED3 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
%         subjectNum
%         trailNum
%         channelNum
%         durationNum
        feature
        label
    end
    
    methods  
        function obj = FeatureDisplay(obj, selArea, showArea)
            %根据GUI界面上的选择来调整数据
            %   此处显示详细说明
            obj = obj.PanelInitianlize(selArea);
            subjectSel = selArea.subjectSel.Value;
            trailSel = selArea.trailSel.Value;
            channelSel = selArea.channelSel.Value;
            featureDispTemp = obj.feature(str2double(subjectSel), str2double(trailSel), str2double(channelSel), :);
            featureDispTemp = reshape(featureDispTemp, [size(featureDispTemp, 3), size(featureDispTemp, 4)]);
            plot(showArea.plotArea, featureDispTemp);
            showArea.subjectSel.Text = subjectSel;
            showArea.trailSel.Text = trailSel;
            showArea.channelSel.Text = channelSel;
            showArea.Duration.Text = num2str(size(obj.feature, 4));            
            
        end
        function obj = LabelDisplay(obj, selArea, showArea)
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            obj = obj.PanelInitianlize(selArea);
            subjectSel = selArea.subjectSel.Value;
            trailSel = selArea.trailSel.Value;
            labelDispTemp = obj.label(str2double(subjectSel), str2double(trailSel));
            showArea.subjectSel.Text = subjectSel;
            showArea.trailSel.Text = trailSel;
            if labelDispTemp == 0
                showArea.labelLamp.Color = 'red';
            else
                showArea.labelLamp.Color = 'green';
            end
        end
        function obj = PanelInitianlize(obj, selArea)
            if ~isempty(obj.feature) && isempty(obj.label)                %特征数据不为空
                for k = 1:size(obj.feature, 1)
                    itemsSub{k} = num2str(k);
                end
                selArea.subjectSel.Items = itemsSub;
                for k = 1:size(obj.feature, 2)
                    itemsTrail{k} = num2str(k);
                end
                selArea.trailSel.Items = itemsTrail;
                for k = 1:size(obj.feature, 3)
                    itempsChannel{k} = num2str(k);
                end
                selArea.channelSel.Items = itempsChannel;
            elseif  ~isempty(obj.feature) && ~isempty(obj.label)        %两个都不为空，选取最小的
                for k = 1:min(size(obj.feature, 1), size(obj.label, 1))
                    itemsSub{k} = num2str(k);
                end
                selArea.subjectSel.Items = itemsSub;
                for k = 1:min(size(obj.feature, 2), size(obj.label, 2))
                    itemsTrail{k} = num2str(k);
                end
                selArea.trailSel.Items = itemsTrail;
                for k = 1:size(obj.feature, 3)
                    itempsChannel{k} = num2str(k);
                end
                selArea.channelSel.Items = itempsChannel;
            end
        end
    end
end

