function net = normalize_cnn_model(net, opts)

	for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            weight_sum = zeros(numel(net.layers{l}.a),1);
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    weight_sum(j) = weight_sum(j) + sum(sum(max(0,net.layers{l}.k{ii}{j})));
                end
            end
            if opts.strong_norm
                for ii = 1 : numel(net.layers{l - 1}.a)
                    for j = 1 : numel(net.layers{l}.a)
                        net.layers{l}.k{ii}{j} = ...
                            net.layers{l}.k{ii}{j} / max(weight_sum);
                    end
                end
            else
                if max(weight_sum) > 1
                    for ii = 1 : numel(net.layers{l - 1}.a)
                        for j = 1 : numel(net.layers{l}.a)
                            net.layers{l}.k{ii}{j} = ...
                                net.layers{l}.k{ii}{j} / max(weight_sum);
                        end
                    end
                end
            end
        end
	end

%     weight_sum = sum(max(0,net.ffW), 2);
%     if opts.strong_norm	
%         net.ffW = net.ffW / max(weight_sum);
%     else
%         if max(weight_sum) > 1
%             net.ffW = net.ffW / max(weight_sum);
%         end
%     end

end

