function cnn=convlifsim(cnn, test_x, test_y, opts)
THRESH = 1;
dt = opts.dt;
performance = [];
% INIT
for l = 1 : numel(cnn.layers)
    cnn.layers{l}.mem = cell(size(cnn.layers{l}.a));
    for j=1:numel(cnn.layers{l}.mem)
        blank_neurons = zeros(size(cnn.layers{l}.a{j}, 1), size(cnn.layers{l}.a{j}, 2), size(test_x, 3));
        cnn.layers{l}.mem{j} = blank_neurons;
        cnn.layers{l}.refrac_end{j} = blank_neurons;        
        cnn.layers{l}.sum_spikes{j} = blank_neurons;
    end
end

cnn.s_o = zeros(10,size(test_x, 3));
cnn.mem_o = zeros(10,size(test_x, 3));
cnn.refrac_end_o = zeros(10,size(test_x, 3));


if strcmp(cnn.layers{end}.type, 'c')
    cnn.sum_fv = zeros(cnn.layers{end}.outputmaps*1, size(test_x, 3));
else
    cnn.sum_fv = zeros(cnn.layers{end-1}.outputmaps*16, size(test_x, 3));
end
for timespan_idx = 1 : numel(opts.timespan)
    timespan = opts.timespan(timespan_idx);
%     input_spikes = zeros([timespan/dt size(test_x,1) size(test_x,2) size(test_x,3)]);
%     for i = 1:size(test_x,1)
%         i
%         for j = 1:size(test_x,2)
%             for m = 1:size(test_x,3)
%                 idxs = randsample(timespan/dt, int32(opts.max_rate*timespan*test_x(i,j,m)));
%                 input_spikes(idxs, i, j, m) = 1;
%             end
%         end
%     end    
%     sum(input_spikes(:,:,:,1), 1)
    for t=dt:dt:timespan
        % create poisson distributed spikes from the input images (for all
        % images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(size(test_x)) * rescale_fac;
        inp_image = spike_snapshot <= test_x; %
        
        % or use exact number of input spikes
%         inp_image = squeeze(input_spikes(int32(t/dt),:,:,:));
        
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
                    cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= THRESH;
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
                %  downsample
                for j = 1 : inputmaps
                    z = convn(cnn.layers{l - 1}.spikes{j}, ones(cnn.layers{l}.scale) / (cnn.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                    z = z(1 : cnn.layers{l}.scale : end, 1 : cnn.layers{l}.scale : end, :);
                    % DO NEURON UPDATE
                    %   Only allow non-refractory neurons to get input
                    z(cnn.layers{l}.refrac_end{j} > t) = 0;
                    %   Add input
                    cnn.layers{l}.mem{j} = cnn.layers{l}.mem{j} + z;
                    %   Check for spiking
                    cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= THRESH;
                    %   Reset
                    cnn.layers{l}.mem{j}(cnn.layers{l}.spikes{j}) = 0;
                    %   Ban updates until....
                    cnn.layers{l}.refrac_end{j}(cnn.layers{l}.spikes{j}) = t + opts.t_ref;              
                    % Store result for analysis later
                    cnn.layers{l}.sum_spikes{j} = cnn.layers{l}.sum_spikes{j} + cnn.layers{l}.spikes{j};                
                end
            end
        end

        %  concatenate all end layer feature maps into vector
        cnn.fv = [];
        for j = 1 : numel(cnn.layers{end}.spikes)
            sa = size(cnn.layers{end}.spikes{j});
            cnn.fv = [cnn.fv; reshape(cnn.layers{end}.spikes{j}, sa(1) * sa(2), sa(3))];
        end
        cnn.sum_fv = cnn.sum_fv + cnn.fv;
        fprintf('.');
%         if(mod(t/dt, 50+1) == 50)
%             fprintf('%2.2f%%\n.', t/((timespan_idx) * timespan) * 100);
%         end
        
        cnn.mem_o = cnn.mem_o + cnn.ffW * cnn.fv;
        cnn.mem_o(cnn.refrac_end_o > t) = 0;
        cnn.spikes_o = cnn.mem_o >= THRESH;
        cnn.mem_o(cnn.spikes_o) = 0;
        cnn.refrac_end_o(cnn.spikes_o) = t + opts.t_ref;  
        cnn.s_o = cnn.s_o + cnn.spikes_o;
        
    end

    for l = 1 : numel(cnn.layers)
        cnn.layers{l}.refrac_end = cell(size(cnn.layers{l}.a));
        for j=1:numel(cnn.layers{l}.mem)
            blank_neurons = zeros(size(cnn.layers{l}.a{j}, 1), size(cnn.layers{l}.a{j}, 2), size(test_x, 3));
            cnn.layers{l}.refrac_end{j} = blank_neurons;      
        end
    end
    cnn.refrac_end_o = zeros(10,size(test_x, 3));
    
    % Get answer
    [~, guess_idx] = max(cnn.s_o);
    [~,   ans_idx] = max(test_y);
    acc = sum(guess_idx==ans_idx)/size(test_x,3)*100;
    performance = [performance; acc];
    %figure(4); plot(performance); drawnow()
    fprintf('Spiking accuracy: %2.2f%%\n', acc);
    
    %  feedforward into output perceptrons
%     cnn.s_o = sigm(cnn.ffW * cnn.sum_fv ...
%                 * 1/(timespan_idx * timespan) ...
%                 * (1/opts.max_rate) ...
%                 + repmat(cnn.ffb, 1, size(cnn.fv, 2)));

end