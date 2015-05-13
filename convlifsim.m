function cnn = convlifsim(cnn, test_x, test_y, opts)
num_examples = size(test_x, 3);
num_classes  = size(test_y, 1);
% Initialize a neuron-based network - needs to be activated to get all the
%   sizes. Shouldn't be an issue after training, unless cleaned.
for l = 1 : numel(cnn.layers)
    outputmaps = numel(cnn.layers{l}.a);
    cnn.layers{l}.mem = cell(1, outputmaps);
    for j=1:outputmaps
        correctly_sized_zeros = zeros(size(cnn.layers{l}.a{j}, 1), ...
            size(cnn.layers{l}.a{j}, 2), num_examples);
        cnn.layers{l}.mem{j} = correctly_sized_zeros;
        cnn.layers{l}.refrac_end{j} = correctly_sized_zeros;        
        cnn.layers{l}.sum_spikes{j} = correctly_sized_zeros;
    end    
end
cnn.sum_fv = zeros(size(cnn.ffW,2), num_examples);
cnn.o_mem        = zeros(num_classes, num_examples);
cnn.o_refrac_end = zeros(num_classes, num_examples);
cnn.o_sum_spikes = zeros(num_classes, num_examples);
cnn.performance  = [];
% Precache answers
[~, ans_idx] = max(test_y);

for t = 0:opts.dt:opts.duration
    % Create poisson distributed spikes from the input images
    %   (for all images in parallel)
    rescale_fac = 1/(opts.dt*opts.max_rate);
    spike_snapshot = rand(size(test_x)) * rescale_fac;
    inp_image = spike_snapshot <= test_x;

    cnn.layers{1}.spikes{1} = inp_image;
    cnn.layers{1}.mem{1} = cnn.layers{1}.mem{1} + inp_image;
    cnn.layers{1}.sum_spikes{1} = cnn.layers{1}.sum_spikes{1} + inp_image;
    inputmaps = 1;
    for l = 2 : numel(cnn.layers)   %  for each layer
        if strcmp(cnn.layers{l}.type, 'c')
            % Convolution layer, output a map for each convolution
            for j = 1 : cnn.layers{l}.outputmaps
                % Sum up input maps
                z = zeros(size(cnn.layers{l - 1}.spikes{1}) - [cnn.layers{l}.kernelsize - 1 cnn.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(cnn.layers{l - 1}.spikes{i}, cnn.layers{l}.k{i}{j}, 'valid');
                end
                % Only allow non-refractory neurons to get input
                z(cnn.layers{l}.refrac_end{j} > t) = 0;
                % Add input
                cnn.layers{l}.mem{j} = cnn.layers{l}.mem{j} + z;
                % Check for spiking
                cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= opts.threshold;
                % Reset
                cnn.layers{l}.mem{j}(cnn.layers{l}.spikes{j}) = 0;
                % Ban updates until....
                cnn.layers{l}.refrac_end{j}(cnn.layers{l}.spikes{j}) = t + opts.t_ref;
                % Store result for analysis later
                cnn.layers{l}.sum_spikes{j} = cnn.layers{l}.sum_spikes{j} + cnn.layers{l}.spikes{j};
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = cnn.layers{l}.outputmaps;
        elseif strcmp(cnn.layers{l}.type, 's')
            %  Subsample by averaging
            for j = 1 : inputmaps
                % Average input
                z = convn(cnn.layers{l - 1}.spikes{j}, ones(cnn.layers{l}.scale) / (cnn.layers{l}.scale ^ 2), 'valid');
                % Downsample
                z = z(1 : cnn.layers{l}.scale : end, 1 : cnn.layers{l}.scale : end, :);
                % Only allow non-refractory neurons to get input
                z(cnn.layers{l}.refrac_end{j} > t) = 0;
                % Add input
                cnn.layers{l}.mem{j} = cnn.layers{l}.mem{j} + z;
                % Check for spiking
                cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= opts.threshold;
                % Reset
                cnn.layers{l}.mem{j}(cnn.layers{l}.spikes{j}) = 0;
                % Ban updates until....
                cnn.layers{l}.refrac_end{j}(cnn.layers{l}.spikes{j}) = t + opts.t_ref;              
                % Store result for analysis later
                cnn.layers{l}.sum_spikes{j} = cnn.layers{l}.sum_spikes{j} + cnn.layers{l}.spikes{j};                
            end
        end
    end

    % Concatenate all end layer feature maps into vector
    cnn.fv = [];
    for j = 1 : numel(cnn.layers{end}.spikes)
        sa = size(cnn.layers{end}.spikes{j});
        cnn.fv = [cnn.fv; reshape(cnn.layers{end}.spikes{j}, sa(1) * sa(2), sa(3))];
    end
    cnn.sum_fv = cnn.sum_fv + cnn.fv;
    
    % Run the output layer neurons
    %   Add inputs multiplied by weight
    impulse = cnn.ffW * cnn.fv;
    %   Only add input from neurons past their refractory point
    impulse(cnn.o_refrac_end >= t) = 0;

    %   Add input to membrane potential
    cnn.o_mem = cnn.o_mem + impulse;
    %   Check for spiking
    cnn.o_spikes = cnn.o_mem >= opts.threshold;
    %   Reset
    cnn.o_mem(cnn.o_spikes) = 0;
    %   Ban updates until....
    cnn.o_refrac_end(cnn.o_spikes) = t + opts.t_ref;
    %   Store result for analysis later
    cnn.o_sum_spikes = cnn.o_sum_spikes + cnn.o_spikes;
    
    % Tell the user what's going on
    if(mod(round(t/opts.dt),round(opts.report_every/opts.dt)) == ...
            0 && (t/opts.dt > 0))
        [~, guess_idx] = max(cnn.o_sum_spikes);
        acc = sum(guess_idx==ans_idx)/size(test_y, 2) * 100;
        fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
        cnn.performance(end+1) = acc;
    else
        fprintf('.');            
    end
end