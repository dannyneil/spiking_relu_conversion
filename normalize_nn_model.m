function [net, norm_constants] = normalize_nn_model(varargin)
    net = varargin{1};
    if(nargin ~= 1)
        dofinal = varargin{2};
    end
    
    if (dofinal)
        endlayer = numel(net.size)-1;
    else
        endlayer = numel(net.size)-2;
    end
    
	for l = 1 : endlayer
        weight_sum = max(sum(max(0,net.W{l}),2));
        norm_constants(l) = 1 ./ weight_sum;
        net.W{l} = net.W{l} ./ weight_sum;
    end
end
