function [nn, scale_factor] = normalize_nn_data(nn, x)
    % Repropagate Activations
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    scale_factor = zeros(1, numel(nn.size)-1);
    for l = 1 : numel(nn.size)-1
        % Repropagate Activations
        nn.testing = 1;
        nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
        nn.testing = 0;        
        % Find the max and rescale
        weight_max = max(max(max(0, nn.W{l})));
        activation_max = max(max(max(0, nn.a{l+1})));
        scale_factor(l) = 1 / max(weight_max, activation_max);
        nn.W{l} = nn.W{l} * scale_factor(l);
    end