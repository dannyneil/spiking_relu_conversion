%% Set up paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
% Load net
load('nn_98.68.mat');
%% Do the different methods
fprintf('\n==================================\n');
fprintf('Ground Truth Normalization Constants:\n');
[norm_nn, norm_constants] = normalize_nn_data_gt(nn, train_x);
fprintf('\nConvNet Way Normalization Constants:\n');
[norm_nn, norm_constants] = normalize_nn_data_conv(nn, train_x);
fprintf('\nFCN Way Normalization Constants:\n');
[norm_nn, norm_constants] = normalize_nn_data_fcn(nn, train_x);