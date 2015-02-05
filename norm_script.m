%% Load path and setup data
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('~/Dropbox/tools/matlab_include'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
clear opts;
%% Test
load('NNDRPSCAN_15_02_02-19_52_22_p_98.68.mat');
[er, bad] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

% Normalize
[net, nc] = normalize_nn_model(nn);
disp(nc);
% Test again
[er, bad] = nntest(net, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
% Show differences
figure(1); clf;
subplot(2,1,1);
dan_viz(nn.W{1}');
colorbar;
subplot(2,1,2);
dan_viz(net.W{1}');
colorbar;
% Save model normalized
nn = net;
save('model_norm_nn.mat','nn');
%% Do Data Normalized
load('NNDRPSCAN_15_02_02-19_52_22_p_98.68.mat');
% Pass in activations
nn.testing = 1;
nn = nnff(nn, train_x, zeros(size(train_x,1), nn.size(end)));
nn.testing = 0;
% Normalize
[er, bad] = nntest(nn, test_x, test_y);
fprintf('Pre-norm Test Accuracy: %2.2f%%.\n', (1-er)*100);
[net, nc] = normalize_nn_data(nn);
% State factors
disp(nc);
[er, bad] = nntest(net, test_x, test_y);
fprintf('Post-norm Test Accuracy: %2.2f%%.\n', (1-er)*100);
% Show
figure(1); clf;
subplot(1,2,1);
dan_viz(nn.W{1}');
colorbar;
subplot(1,2,2);
dan_viz(net.W{1}');
colorbar;
% Save
%   Reset nn
net.testing = 1;
net = nnff(net, train_x(1:10,:), zeros(10, net.size(end)));
net.testing = 0;
nn = net;
save('data_norm_nn.mat','nn');