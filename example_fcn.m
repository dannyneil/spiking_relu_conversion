%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
% Initialize net
nn = nnsetup([784 1200 1200 10]);
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between [-0.1 0.1]
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
    nn.vW{i - 1} = zeros(size(nn.W{i-1}));
end
%% ReLU Train
% Set up learning constants
nn.activation_function = 'relu';
nn.output ='relu';
nn.learningRate = 1;
nn.momentum = 0.5;
nn.dropoutFraction = 0.5;
nn.learn_bias = 0;
opts.numepochs =  15;
opts.batchsize = 100;
% Train - takes about 15 seconds per epoch on my machine
nn = nntrain(nn, train_x, train_y, opts);
% Test - should be 98.62% after 15 epochs
[er, train_bad] = nntest(nn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
%% Spike-based Testing of Fully-Connected NN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.035;
t_opts.report_every = 0.001;
t_opts.max_rate     =   200;
nn = nnlifsim(nn, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Data-normalize the NN
[norm_nn, norm_constants] = normalize_nn_data(nn, train_x);
fprintf('NN normalized.\n');
for idx=1:numel(norm_constants)
    fprintf('Normalization Factor for Layer %i: %3.5f\n',idx, norm_constants(idx));
end
fprintf('NN normalized.\n');
%% Test the Data-Normalized NN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.035;
t_opts.report_every = 0.001;
t_opts.max_rate     =  1000;
norm_nn = nnlifsim(norm_nn, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Show the difference
figure(1); clf;
plot(t_opts.dt:t_opts.dt:t_opts.duration, norm_nn.performance);
hold on; grid on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn.performance);
legend('Normalized Net, Default Params', 'Unnormalized Net, Best Params');
ylim([90 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');