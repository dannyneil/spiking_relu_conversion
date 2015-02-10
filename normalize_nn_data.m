function [nn, applied_factor] = normalize_nn_data(nn, x)
    % Repropagate Activations
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    scale_factor = zeros(1, numel(nn.size)-1);
    previous_factor = 1;
    for l = 1 : numel(nn.size)-1 
        % Find the max and rescale
        weight_max = max(max(max(0, nn.W{l})));
        activation_max = max(max(max(0, nn.a{l+1})));
        scale_factor = max(weight_max, activation_max);
        applied_factor(l) = previous_factor / scale_factor;
        nn.W{l} = nn.W{l} * applied_factor(l);
        previous_factor = scale_factor;
    end