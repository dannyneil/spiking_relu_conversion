function [nn, current_factor] = normalize_nn_data(nn, x)
    % Get Activations
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;    
    
    % Normalize based on activations
    previous_factor = 1;    
    for l = 1 : numel(nn.size)-1
        weight_max = max(max(max(0, nn.W{l})));
        activation_max = max(max(max(0, nn.a{l+1})));
        scale_factor = max(weight_max, activation_max);
        current_factor(l) =  scale_factor/ previous_factor;
        nn.W{l} = nn.W{l} / current_factor(l);
        previous_factor = current_factor(l);
    end
end
