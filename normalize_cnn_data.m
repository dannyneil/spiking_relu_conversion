function [net, factor_log] = normalize_cnn_data(net, train_x)
    % Propagate data through the net
    net = cnnff(net, train_x);
        
    previous_factor = 1;
    factor_log = nan(1, numel(net.layers));
	for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            max_weight = 0;
            max_activation = 0;
            for i = 1:size(net.layers{l}.k)
                for j = 1:size(net.layers{l}.k{i})
                    max_weight = max(max_weight, max(max(net.layers{l}.k{i}{j})));
                end
            end
            for i = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    max_activation = max(max_activation, max(max(max(net.layers{l}.a{i}))));
                end
            end
            scale_factor = max(max_weight, max_activation);
            current_factor = scale_factor / previous_factor;
            for ii = 1 : numel(net.layers{l - 1}.a)
                for j = 1 : numel(net.layers{l}.a)
                    net.layers{l}.k{ii}{j} = ...
                        net.layers{l}.k{ii}{j} / current_factor;
                end
            end
            factor_log(l) = 1 / current_factor;
            previous_factor =  current_factor;
        end
    end

%     max_weight = max(max(net.ffW));
%     max_activation = max(max(net.o));
%     final_factor = max(max_weight, max_activation);
%     current_factor =  final_factor / previous_factor;
%     net.ffW = net.ffW / current_factor;
%     factor_log(end+1) = 1 / current_factor;
end

