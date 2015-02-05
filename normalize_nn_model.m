function [net, norm_constants] = normalize_nn_model(net)
	for l = 1 : numel(net.size)-1
        weight_sum = max(sum(max(0,net.W{l}),2));
        norm_constants(l) = weight_sum;
        net.W{l} = net.W{l} ./ weight_sum;
    end
end
