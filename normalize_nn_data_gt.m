function [nn, factor] = normalize_nn_data_gt(nn, x)
    % Repropagate Activations
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    fprintf('Activations before scaling:\n');
    for l = 1 : numel(nn.size)
        activation_max = max(max(max(0, nn.a{l})));
        fprintf('%2.5f\n', activation_max);
    end
    fprintf('Weights before scaling:\n');
    for l = 1 : numel(nn.size)-1
        weight_max = max(max(max(0, nn.W{l})));
        fprintf('%2.5f\n', weight_max);
    end
    % Normalize based on activations    
    fprintf('Scaling Factors:\n');
    previous_factor = 1;
    for l = 1 : numel(nn.size)-1
        % Repropagate Activations
        nn.testing = 1;
        nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
        nn.testing = 0;        
        % Find the max and rescale
        weight_max = max(max(max(0, nn.W{l})));
        activation_max = max(max(max(0, nn.a{l+1})));
        factor = max(weight_max, activation_max);
        nn.W{l} = nn.W{l} * 1 / factor;
        fprintf('%2.5f\n',1/factor);
    end

% Repropagate Activations
nn.testing = 1;
nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
nn.testing = 0;
fprintf('Activations after scaling:\n');
for l = 1 : numel(nn.size)
    activation_max = max(max(max(0, nn.a{l})));
    fprintf('%2.5f\n', activation_max);
end
fprintf('Weights after scaling:\n');
for l = 1 : numel(nn.size)-1
    weight_max = max(max(max(0, nn.W{l})));
    fprintf('%2.5f\n', weight_max);
end