close all
clearvars
clc 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               PARAMETERS                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prs = containers.Map;
% Simulation paramters
    prs('dt') = 0.00004;  % 0.04 ms
% Network parameters 
    prs('N') = 1000;  %Number of neurons
    prs('p') = 0.1; %Set the network sparsity 
% Neuron parameters
    prs('tref') = 0.002; %Refractory time constant in seconds 2ms
    prs('tm') = 0.01; %Membrane time constant 10ms (original)
    prs('vreset') = -65; %Voltage reset mV
    prs('vpeak') = -40; %Voltage peak. mV
    prs('BIAS') = prs('vpeak'); %Set the BIAS current, can help decrease/increase firing rates.  0 is fine. 
% Synaptic parameters
    prs('td') = 0.02; % synaptic constant
    prs('tr') = 0.002; % synaptic constant 
% Learning Parameters
    prs('Q') = 10;  % Learned weights factor
    prs('G') = 0.04;  % Chaotic weights factor
    prs('alpha') = prs('dt')*0.1 ; %Sets the rate of weight change, too fast is unstable, too slow is bad as well. 
    prs('step') = 50;  % FORCE update every 50*dt=2.5ms
% Supervisor
    prs('t_start_FORCE') = 1;
    prs('t_stop_FORCE') = 2.5;
    prs('T') = 5;

save('prs.mat', 'prs')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            BIRD SONG WITH HDTS                          %
%                     Gamma - Varying Synaptic constants                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clearvars
clc 
load('prs.mat')
prs('dt') = 0.00004;
dt = prs('dt');


display_error = 0;
trials = 2;  % 10 in original paper

% Supervisor signal
[wave,fs] = audioread('ZebraFinch.wav'); 
D = 2;  % Decimate by D
wave = decimate(wave,D);
fs = fs/D;  % New sampling freq.
[s,f,t] = spectrogram(wave, 512, 511, 512,fs);  % Generate spectrogram
dt_song = t(2)-t(1);  % Update dt

% PSD
zx = log(abs(s).^2);  % PSD
zx = zx/norm(zx, 'fro');
zx(isnan(zx)==1)=0; 
song_time = max(size(zx))*dt;

% Plot spectogram
figure
surf(t, f, zx, 'edgecolor','none')
view(2)
colorbar

% HDTS signal
m2 = 500;    
TT = 5; % s
TT = 1000*TT; % ms
dt = 0.04;
temp1 = abs( sin(m2*pi*((1:1:TT/dt)*dt)/TT) );  % Fit m2 pulses in TT ms
z2 = zeros(m2, numel(temp1));
for qw = 1:1:m2
    z2(qw,:) = temp1.*((1:1:TT/dt)*dt<qw*TT/m2).*((1:1:TT/dt)*dt>(qw-1)*TT/m2);
end
prs('z2') = z2;  % HDTS signal repeats m2 pulses every TT seconds
TT = 5;  % s
dt = prs('dt');

% Update parameters (prs)
prs('T') = TT;
prs('t_start_FORCE') = 1;
prs('t_stop_FORCE') = 0.5*prs('T');
prs('nt') = round(prs('T')/prs('dt'));
prs('zx') = zx;  % Supervisor
save('prs.mat', 'prs')
% Gamma distribution parameters
shape = 3;
scale = 0.025;


Ns = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
train_error_het_gamma = zeros(trials, numel(Ns));
test_error_het_gamma = zeros(trials, numel(Ns));
train_error_het = zeros(trials, numel(Ns));
test_error_het = zeros(trials, numel(Ns));
train_error_hom = zeros(trials, numel(Ns));
test_error_hom = zeros(trials, numel(Ns));
for N_index = 1:numel(Ns)
    
    N = Ns(N_index);    
    prs('N') = N;
    
    % HETEROGENEOUS GAMMA
    type = ['GAMMA N=', num2str(prs('N'))];
    for n = 1:trials
        seed = n;

        lif_net = LIFFORCESONG_HDTS(seed, prs, N);
        
        % synaptic constants   
        td = gamrnd(shape, scale, N, 1);  
        tr =  0.002; 

        lif_net.td = td;
        lif_net.tr = tr;

        lif_net.reset(lif_net, seed)
        lif_net.run_net(lif_net)
        lif_net.get_error(lif_net, display_error)

        train_error_het_gamma(n, N_index) = lif_net.train_error;    
        test_error_het_gamma(n, N_index) =  lif_net.test_error;

        lif_net.plot_train_test(lif_net, type)
    end    
    [train_time, test_time, zx_train_gamma, zx_test_gamma, zx_train_hat_gamma, zx_test_hat_gamma] = lif_net.get_output(lif_net);    
    error_disp = mean(test_error_het_gamma(:,N_index), 1);
    disp(['N=', num2str(N), ' Testing Error HDTS Hetergeneous Gamma = ', num2str(error_disp)])
    if N==4000
        save('heterogeneous_gamma_4000.mat', 'lif_net');
    end
    
    % HETEROGENEOUS
    type = 'HETEROGENEOUS ';
    for n = 1:trials
        seed = n;

        lif_net = LIFFORCESONG_HDTS(seed, prs, N);

        td = zeros(N, 1);  % just for parfor sake
        tr = zeros(N, 1);  % just for parfor sake
        td(1:N, 1) = 0.1*ones(N, 1); % synaptic constant
        tr(1:N, 1) = 0.002*ones(N, 1); % synaptic constant 
        td(N/2+1:end, 1) = 0.02*ones(N/2, 1); % synaptic constant
        tr(N/2+1:end, 1) = 0.002*ones(N/2, 1); % synaptic constant  

        lif_net.td = td;
        lif_net.tr = tr;

        lif_net.reset(lif_net, seed)
        lif_net.run_net(lif_net)
        lif_net.get_error(lif_net, display_error)

        train_error_het(n, N_index) = lif_net.train_error;    
        test_error_het(n, N_index) = lif_net.test_error;

        lif_net.plot_train_test(lif_net, type)
    end
    [~, ~, zx_train_het, zx_test_het, zx_train_hat_het, zx_test_hat_het] = lif_net.get_output(lif_net);
    error_disp = mean(test_error_het(:,N_index), 1);
    disp(['N=', num2str(N), ' Testing Error HDTS Hetergeneous = ', num2str(error_disp)])
    if N==4000
        save('heterogeneous_4000.mat', 'lif_net');
    end
    
    % HOMOGENEOUS
    type = 'HOMOGENEOUS ';
    for n = 1:trials
        seed = n;

        lif_net = LIFFORCESONG_HDTS(seed, prs, N);

        td = 0.02; % synaptic constant
        tr = 0.002; % synaptic constant 

        lif_net.td = td;
        lif_net.tr = tr;

        lif_net.reset(lif_net, seed)
        lif_net.run_net(lif_net)
        lif_net.get_error(lif_net, display_error)

        train_error_hom(n, N_index) = lif_net.train_error;    
        test_error_hom(n, N_index) = lif_net.test_error;

        lif_net.plot_train_test(lif_net, type)
    end
    [~, ~, zx_train_hom, zx_test_hom, zx_train_hat_hom, zx_test_hat_hom] = lif_net.get_output(lif_net);    
    error_disp = mean(test_error_hom(:,N_index), 1);
    disp(['N=', num2str(N), ' Testing Error HDTS Homogeneous = ', num2str(error_disp)])
    if N==4000
        save('homogeneous_4000.mat', 'lif_net');
    end
    
end


save('FORCE_het_robust.mat', 'Ns', ...
            'train_error_het_gamma', 'test_error_het_gamma', ...
            'train_error_het', 'test_error_het', ...
            'train_error_hom', 'test_error_hom')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            BIRD SONG WITH HDTS                          %
%                             Plot Spectrograms                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load('prs.mat')
dt = 4e-5;
song_time = 1.1828;

% Plot reconstructed spectrograms
dt_rec = 10*dt;  % ms (simulation was done at 0.04 but recordings every 10 steps
start_test = song_time - rem(2.5, song_time); % A full song takes song_time seconds 
start_test_i = ceil(start_test / dt_rec);
end_test_i = start_test_i + floor(song_time / dt_rec);

% Homogeneous
clc
clearvars -except start_test_i end_test_i dt_rec
load(['homogeneous_4000.mat'])

figure
surf(lif_net.test_curr', 'edgecolor','none')
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
xlim([start_test_i, end_test_i])  % Test happens between
x_ticks = linspace(start_test_i,end_test_i, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
view(2)
colorbar
title(['Homogeneous Test zx - Test Error: ', num2str(lif_net.test_error)])  


% Heterogeneous
clc
clearvars -except start_test_i end_test_i dt_rec
load(['heterogeneous_4000.mat'])

figure
surf(lif_net.test_curr', 'edgecolor','none')
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
xlim([start_test_i, end_test_i])  % Test happens between
x_ticks = linspace(start_test_i,end_test_i, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
view(2)
colorbar
title(['Heterogeneous Double Test zx - Test Error: ', num2str(lif_net.test_error)])  


% Heterogeneous Gamma
clc 
clearvars -except start_test_i end_test_i dt_rec
load(['heterogeneous_gamma_4000.mat'])

figure
surf(lif_net.test_curr', 'edgecolor','none')
xlabel('Time (s)')
ylabel('Frequency (Hz)')
xlim([start_test_i, end_test_i])  % Test happens between
x_ticks = linspace(start_test_i,end_test_i, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
view(2)
colorbar
title(['Heterogeneous Gamma Test zx - Test Error: ', num2str(lif_net.test_error)])  


% Original Spectrogram
figure
surf(lif_net.zx, 'edgecolor','none')
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
xticks([0 5000 10000 15000 20000 25000 30000])
xticklabels({['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2']})
view(2)
colorbar
title('Original')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            BIRD SONG WITH HDTS                          %
%                                Plot Spikes                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% load('prs.mat')
dt = 4e-5;
song_time = 1.1828;

% Plot reconstructed spectrograms
dt_rec = 10*dt;  % ms (simulation was done at 0.04 but recordings every 10 steps
start_test = song_time - rem(2.5, song_time); % A full song takes song_time seconds 
start_test_i = ceil(start_test / dt_rec);
end_test_i = start_test_i + floor(song_time / dt_rec);

% Homogeneous
clc
clearvars -except start_test_i end_test_i dt_rec start_test song_time
load(['homogeneous_4000.mat'])

figure
plot(lif_net.tspike(:, 2), lif_net.tspike(:, 1),'k.', 'Markersize', 4)
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylim([0, 50.5])
xlim([2.5+start_test, 2.5+start_test+song_time])  % Test happens between
x_ticks = linspace(2.5+start_test, 2.5+start_test+song_time, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
title(['Homogeneous Test Spikes - Test Error: ', num2str(lif_net.test_error)])  


% Heterogeneous
clc
clearvars -except start_test_i end_test_i dt_rec start_test song_time
load(['heterogeneous_4000.mat'])

figure
plot(lif_net.tspike(:, 2), lif_net.tspike(:, 1),'k.', 'Markersize', 4)
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylim([0, 50.5])
xlim([2.5+start_test, 2.5+start_test+song_time])  % Test happens between
x_ticks = linspace(2.5+start_test, 2.5+start_test+song_time, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
title(['Heterogeneous Double Spikes - Test Error: ', num2str(lif_net.test_error)])  


% Heterogeneous Gamma
clc  
clearvars -except start_test_i end_test_i dt_rec start_test song_time
load(['heterogeneous_gamma_4000.mat'])

figure
plot(lif_net.tspike(:, 2), lif_net.tspike(:, 1),'k.', 'Markersize', 4)
xlabel('Time (s)')
ylabel('Frequency (Hz)')
ylim([0, 50.5])
xlim([2.5+start_test, 2.5+start_test+song_time])  % Test happens between
x_ticks = linspace(2.5+start_test, 2.5+start_test+song_time, 7);
xticks(x_ticks)
xticklabels({string(round((x_ticks-start_test_i)*dt_rec, 1))})
title(['Heterogeneous Gamma Spikes - Test Error: ', num2str(lif_net.test_error)])  



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            BIRD SONG WITH HDTS                          %
%                                Plot Error                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clc

load('FORCE_het_robust.mat')

f = figure();
hold on
options.handle = f;
options.alpha = 0.3;
options.line_width = 2;
options.error = 'std';
options.x_axis = Ns;
options.color_area = hex2rgb('#1F77B4');    % Blue theme
options.color_line = hex2rgb('#1F77B4');
options.displayname = 'Heterogeneous Gamma';
plot_areaerrorbar(train_error_het_gamma, options);
options.color_area = hex2rgb('#FF7F0E');    % Orange theme
options.color_line = hex2rgb('#FF7F0E');
options.displayname = 'Heterogeneous Double';
plot_areaerrorbar(train_error_het, options);
options.color_area = hex2rgb('#2CA02c');    % Green theme
options.color_line = hex2rgb('#2CA02c');
options.displayname = 'Homogeneous';
plot_areaerrorbar(train_error_hom, options);
legend('show')
ylabel('Log-MSE')
xlabel('Number of neurons')
title('Train error')
grid on
grid minor


f = figure();
hold on
options.handle = f;
options.alpha = 0.3;
options.line_width = 2;
options.error = 'std';
options.x_axis = Ns;
options.color_area = hex2rgb('#1F77B4');    % Blue theme
options.color_line = hex2rgb('#1F77B4');
options.displayname = 'Heterogeneous Gamma';
plot_areaerrorbar(test_error_het_gamma, options);
options.color_area = hex2rgb('#FF7F0E');    % Orange theme
options.color_line = hex2rgb('#FF7F0E');
options.displayname = 'Heterogeneous Double';
plot_areaerrorbar(test_error_het, options);
options.color_area = hex2rgb('#2CA02c');    % Green theme
options.color_line = hex2rgb('#2CA02c');
options.displayname = 'Homogeneous';
plot_areaerrorbar(test_error_hom, options);
legend('show')
ylabel('Log-MSE')
xlabel('Number of neurons')
title('Test error')
grid on
grid minor


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            BIRD SONG WITH HDTS                          %
%                              Run gridsearch                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ns = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
for n=1:numel(Ns)
    nb_neurons = Ns(n);    
    GRIDSEARCH(nb_neurons)    
end


