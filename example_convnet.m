%% Train an example ConvNet to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 1);
load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000)) / 255;
test_x = double(reshape(test_x',28,28,10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');
% Initialize net
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);
% Set the activation function to be a ReLU
cnn.act_fun = @(inp)max(0, inp);
% Set the derivative to be the binary derivative of a ReLU
cnn.d_act_fun = @(forward_act)double(forward_act>0);
%% ReLU Train
% Set up learning constants
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs =  5;
opts.learn_bias = 0;
opts.dropout = 0.0;
cnn.first_layer_dropout = 0;
% Train - takes about 199 seconds per epoch on my machine
cnn = cnntrain(cnn, train_x, train_y, opts);
% Test
[er, bad] = cnntest(cnn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
%% Spike-based Testing of a ConvNet
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.040;
t_opts.report_every = 0.001;
t_opts.max_rate     =   400;
cnn = convlifsim(cnn, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Data-normalize the CNN
[norm_convnet, norm_constants] = normalize_cnn_data(cnn, train_x);
for idx=1:numel(norm_constants)
    fprintf('Normalization Factor for Layer %i: %3.5f\n',idx, norm_constants(idx));
end
fprintf('ConvNet normalized.\n');
%% Test the Data-Normalized CNN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.040;
t_opts.report_every = 0.001;
t_opts.max_rate     =  1000;
norm_convnet = convlifsim(norm_convnet, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Show the difference
figure(1); clf;
plot(t_opts.dt:t_opts.dt:t_opts.duration, norm_convnet.performance);
hold on; grid on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, cnn.performance);
legend('Normalized ConvNet, Default Params', 'Unnormalized ConvNet');
ylim([00 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');