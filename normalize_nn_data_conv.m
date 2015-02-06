function [nn, factor] = normalize_nn_data_conv(nn, x)
    % Repropagate Activations
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;        
    % Normalize based on activations    
    fprintf('Scaling Factors:\n');
    previous_factor = 1;    
    for l = 1 : numel(nn.size)-1
        % Find the max and rescale
        weight_max = max(max(max(0, nn.W{l})));
        activation_max = max(max(max(0, nn.a{l+1})));
        factor = max(weight_max, activation_max);
        current_factor = previous_factor / factor;
        nn.W{l} = nn.W{l} * current_factor;
        fprintf('%2.5f\n', current_factor);
        previous_factor = factor;
    end

% Repropagate Activations
nn.testing = 1;
nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
nn.testing = 0;
fprintf('Activations after scaling:\n');
for l = 1 : numel(nn.size)
    activation_max = max(max(max(0, nn.a{l})));
    fprintf('%2.5f\n',activation_max);
end
fprintf('Weights after scaling:\n');
for l = 1 : numel(nn.size)-1
    weight_max = max(max(max(0, nn.W{l})));
    fprintf('%2.5f\n', weight_max);
end