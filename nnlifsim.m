function nn=nnlifsim(nn, test_x, test_y, opts)
dt = opts.dt;
nn.performance = [];
num_examples = size(test_x,1);

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;        
    nn.layers{l}.sum_spikes = blank_neurons;
end

% Precache answers
[~,   ans_idx] = max(test_y');

% Time-stepped simulation
for t=dt:dt:opts.duration
        % Create poisson distributed spikes from the input images
        %   (for all images in parallel)
        rescale_fac = 1/(dt*opts.max_rate);
        spike_snapshot = rand(size(test_x)) * rescale_fac;
        inp_image = spike_snapshot <= test_x;

        nn.layers{1}.spikes = inp_image;
        nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
        for l = 2 : numel(nn.size)
            % Get input impulse from incoming spikes
            impulse = nn.layers{l-1}.spikes*nn.W{l-1}';
            % Add input to membrane potential
            nn.layers{l}.mem = nn.layers{l}.mem + impulse;
            % Check for spiking
            nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold;
            % Reset
            nn.layers{l}.mem(nn.layers{l}.spikes) = 0;
            % Ban updates until....
            nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + opts.t_ref;
            % Store result for analysis later
            nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;            
        end
        if(mod(round(t/dt),round(opts.report_every/dt)) == round(opts.report_every/dt)-1)
            [~, guess_idx] = max(nn.layers{end}.sum_spikes');
            acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
            fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
            nn.performance(end+1) = acc;
        else
            fprintf('.');            
        end
end
    
    
% Get answer
[~, guess_idx] = max(nn.layers{end}.sum_spikes');
acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
fprintf('\nFinal spiking accuracy: %2.2f%%\n', acc);
end
