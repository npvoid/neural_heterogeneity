classdef LIFFORCESONG_HDTS < handle
    properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                               PARAMETERS                                %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Simulation paramters
            dt  % sim. time step in seconds
        % Network parameters 
            N  % Number of neurons
            p  % Probability of connection
        % Neuron parameters
            tref  % Refractory time constant in seconds
            tm  % Membrane time constant in seconds
            vreset  % Voltage reset mV
            vpeak  % Voltage peak mV
            BIAS  % BIAS current, can help decrease/increase firing rates.  0 is fine. mV
        % Synaptic parameters
            td  % synaptic decay constant in seconds
            tr  % synaptic rise constant in seconds
        % Learning Parameters
            Q  % Learned weights factor
            G  % Chaotic weights factor
            WE2 % HDTS weight factor
            alpha  % Sets the rate of weight change, too fast is unstable, too slow is bad as well.  
            Pinv  % Correlation weight matrix for RLMS
            step  % FORCE update step in seconds
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                               SUPERVISOR                                %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            t_start_FORCE  % Time to start FORCE in seconds
            t_stop_FORCE  % Time to stop FORCE in seconds
            T  % Time to stop testing in seconds (same as total time)
            imin % Index to start FORCE
            icrit % Index to stop FORCE
            nt  % Number of timesteps
            zx  % Supervisor
            z2  % HDTS signal
            k  % Dimensionality of supervisor
            m2  % Dimensionality of HDTS
            nt_HDTS  % Number of time steps in HDTS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                                 VARS                                    %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Neurons
            v  % Membrane potential vector
            tlast  % Vector is used to set the refractory times 
            tspike  % Storage variable for spike times
        % Synaptic filters
            IPSC  % post synaptic current storage variable 
            h  % Storage variable for filtered firing rates
            r  % second storage variable for filtered rates 
            hr  % Third variable for filtered rates
            JD  % storage variable required for each spike time 
        % Weights
            E  % eta vector
            E2  % eta vector for HDTS
            OMEGA  % The initial weight matrix with fixed random weights  
            BPhi  % The initial matrix that will be learned by FORCE method
            kill  % Dead Neurons
        % Recording vars
            current  % Storage variable for output current/approximant 
            orig  % original downsampled
            z  % Approximant 
            RECB % Storage matrix for the synaptic weights (a subset of them) 
        % Plotting vars
            ns  % Number of spikes, counts during simulation  
            time  % Time vector for plotting
            train_time  % Time vector for plotting training time
            test_time  % Time vector for plotting testing time
            train_curr  % Training approximant
            test_curr  % Testing approximant
            train_zx  % Training supervisor
            test_zx  % Testing supervisor
            train_error  % Train error
            test_error  % Test error
    end
    methods
        function self = LIFFORCESONG_HDTS(seed, prs, N)
            rng(seed);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                               PARAMETERS                                %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Simulation paramters
                self.dt = prs('dt');  % 0.05 ms
            % Network parameters 
                self.N = N;  %Number of neurons
                self.p = prs('p'); %Set the network sparsity 
            % Neuron parameters
                self.tref = prs('tref'); %Refractory time constant in seconds 2ms
                self.tm = prs('tm'); %Membrane time constant 10ms (original)
                self.vreset = prs('vreset'); %Voltage reset mV
                self.vpeak = prs('vpeak'); %Voltage peak. mV
                self.BIAS = prs('BIAS'); %Set the BIAS current, can help decrease/increase firing rates.  0 is fine. 
            % Synaptic parameters
                self.td = prs('td'); % synaptic constant
                self.tr = prs('tr'); % synaptic constant 
            % Learning Parameters
                self.Q = prs('Q');  % Learned weights factor
                self.G = prs('G');  % Chaotic weights factor
                self.WE2 = 8*self.Q;  % HDTS factor
                self.alpha = prs('alpha') ; %Sets the rate of weight change, too fast is unstable, too slow is bad as well.  
                self.Pinv = eye(self.N)*self.alpha; %initialize the correlation weight matrix for RLMS
                self.step = prs('step');  % FORCE update every 50*dt=2.5ms
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                               SUPERVISOR                                %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                self.t_start_FORCE = prs('t_start_FORCE');
                self.t_stop_FORCE = prs('t_stop_FORCE');
                self.T = prs('T');
                self.imin = round(self.t_start_FORCE/self.dt); % start FORCE after t_start_FORCE seconds
                self.icrit = round(self.t_stop_FORCE/self.dt); % stop FORCE after t_end seconds
                self.nt = round(self.T/self.dt);  % Number of timesteps
                self.zx = prs('zx');  % Supervisor
                self.z2 = prs('z2');  % HDTS signal
                self.k = min(size(self.zx));  % Dimensionality of supervisor
                self.m2 = min(size(self.z2));
                self.nt_HDTS = max(size(self.zx));
%                 if self.nt_HDTS ~= max(size(self.zx))
%                     error('HDTS and Supervisor must have same length')
%                 end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                          VARS INITIALISATION                            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Init neurons
                self.v = self.vreset + rand(self.N,1)*(30-self.vreset); % Initialize neuronal voltage with random distribtuions
                self.tlast = zeros(self.N,1); %This vector is used to set  the refractory times 
                self.tspike = zeros(4*self.nt,2); %Storage variable for spike times
            % Init synaptic filters
                self.IPSC = zeros(self.N,1); % post synaptic current storage variable 
                self.h = zeros(self.N,1); %Storage variable for filtered firing rates
                self.r = zeros(self.N,1); %second storage variable for filtered rates 
                self.hr = zeros(self.N,1); % Third variable for filtered rates
                self.JD = 0*self.IPSC; %storage variable required for each spike time 
            % Init weights
                self.E = self.Q*(2*rand(self.N, self.k)-1);  %n
                self.E2 = self.WE2*(2*rand(self.N, self.m2)-1); %HDTS input
                self.OMEGA =  self.G*(randn(self.N,self.N)).*(rand(self.N,self.N)<self.p)/(sqrt(self.N)*self.p); % The initial weight matrix with fixed random weights  
                self.BPhi = zeros(self.N, self.k); % The initial matrix that will be learned by FORCE method
                for i = 1:1:self.N     %set the row average weight to be zero, explicitly. (average input weight to each neuron??)
                    QS = find(abs(self.OMEGA(i,:))>0);
                    self.OMEGA(i, QS) = self.OMEGA(i, QS) - sum(self.OMEGA(i, QS))/length(QS);
                end
                self.kill = ones(size(self.v));
            % Init recording vars 
                self.ns = 0;  % Number of spikes, counts during simulation  
                self.current = zeros(floor(self.nt/10),self.k);  %storage variable for output current/approximant 
                self.orig = zeros(floor(self.nt/10),self.k);  %storage variable for output current/approximant 
                self.z = zeros(self.k, 1);  %Initialize the approximant                 
                self.time = 1:1:self.nt;  % Time vector for plotting
                self.RECB = zeros(floor(self.nt/10),10);  % Storage matrix for the synaptic weights (a subset of them) 
        end
    end
    methods (Static)
        function run_net(self)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                             RUN SIMULATION                              %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            qq = 1;  % HDTS counter
            for i = 1:1:self.nt 

                I = self.IPSC + self.E*self.z + self.E2*self.z2(:, qq) + self.BIAS; % Neuronal Current 

                dv = (self.dt*i>self.tlast + self.tref).*(-self.v+I) ./ self.tm; %Voltage equation with refractory period 
                self.v = self.v + self.dt*(dv);

                index = find(self.v>=self.vpeak);  %Find the neurons that have spiked 

                %Store spike times, and get the weight matrix column sum of spikers 
                if ~isempty(index)
                    self.JD = sum(self.OMEGA(:,index),2); %compute the increase in current due to spiking  
                    self.tspike(self.ns+1:self.ns+length(index),:) = [index,0*index+self.dt*i];
                    self.ns = self.ns + length(index);  % total number of psikes so far
                end

                self.tlast = self.tlast + (self.dt*i - self.tlast).*(self.v>=self.vpeak);  %Used to set the refractory period of LIF neurons 

                % Code if the rise time is 0, and if the rise time is positive 
                if self.tr == 0  
                    self.IPSC = self.IPSC.*exp(-self.dt ./ self.td)+   self.JD.*(~isempty(index)) ./ (self.td);
                    self.r = self.r.*exp(-self.dt ./ self.td) + (self.v>=self.vpeak) ./ self.td;
                else
                    self.IPSC = self.IPSC.*exp(-self.dt ./ self.tr) + self.h*self.dt;
                    self.h = self.h.*exp(-self.dt ./ self.td) + self.JD.*(~isempty(index)) ./ (self.tr .* self.td);  %Integrate the current

                    self.r = self.r .* exp(-self.dt ./ self.tr) + self.hr*self.dt; 
                    self.hr = self.hr.*exp(-self.dt ./ self.td) + (self.v>=self.vpeak) ./ (self.tr.*self.td);
                end

                %Implement RLMS with the FORCE method 
                self.z = self.BPhi' * self.r; %approximant 
                err = self.z - self.zx(:, qq); %error 
                % RLMS 
                if mod(i, self.step)==1 
                    if i > self.imin 
                        if i < self.icrit 
                            cd = self.Pinv * self.r;
                            self.BPhi = self.BPhi - (cd*err');
                            self.Pinv = self.Pinv -((cd)*(cd'))/( 1 + (self.r')*(cd));
                        end 
                    end 
                end

                self.v = self.v + (30 - self.v).*(self.v>=self.vpeak);

                self.v = self.v + (self.vreset - self.v).*(self.v>=self.vpeak); %reset with spike time interpolant implemented.  
                
                if i/self.nt > 0.5
                    self.v = self.v .* self.kill;
                end
                
                if mod(i, 10)==0
                    ii = floor(i/10)+1;
                    self.current(ii,:) = self.z;
                    self.orig(ii, :) = self.zx(:, qq);
                    self.RECB(ii,:) = self.BPhi(1:10); 
                end
                
                if qq>=self.nt_HDTS
                    qq = 0;
                end
                qq = qq + 1;
                
                if mod(i,round(1/self.dt))==1
%                     clc
                    disp( ['t = ', num2str(round(i*self.dt)), 's'] )
                end
                
                
%                 if mod(i,round(0.5/self.dt))==1                    
%                     drawnow
%                     figure(5)
%                     plot(self.dt*(1:1:size(self.RECB, 1)), self.RECB(:,1:10),'.')
%                 end

            end

 
        end
        
        function get_error(self, display)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                         TRAIN AND TEST ERRORS                           %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % FORCE training stops at 2.5s, total time 5s
            M = floor(size(self.current, 1)/2);
            self.train_curr = self.current(1:1:M, :);
            self.train_zx = self.orig(1:1:M, :);
            
            self.test_curr = self.current(M:1:end, :);
            self.test_zx = self.orig(M:1:end, :);
            
            self.train_error = log10( 0.5 * sum(sum((self.train_curr-self.train_zx).^2)) / numel(self.train_zx) );
            self.test_error = log10( 0.5 * sum(sum((self.test_curr-self.test_zx).^2)) / numel(self.test_zx) );
            
            if display
                disp(['Training Error = ', num2str(self.train_error)])
                disp(['Testing Error = ', num2str(self.test_error)])
            end
        end
        
               
        function [train_time, test_time, zx_train, zx_test, zx_train_hat, zx_test_hat] = get_output(self)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                         TRAIN AND TEST ERRORS                           %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            self.get_error(self, 0)
            
            train_time = self.train_time;
            test_time = self.test_time;
            
            zx_train_hat = self.train_curr;
            zx_test_hat = self.test_curr;

            zx_train = self.zx';
            zx_test = self.zx';              
        end
        
        function plot_train_test(self, type, t, f)
            if nargin == 2
                self.get_error(self, 0)
                figure
                surf(self.zx, 'edgecolor','none')
                xlabel('Time (s)')
                ylabel('Frequency (Hz)')
                view(2)
                colorbar
                title('Original')
%                 figure
%                 surf(self.train_curr', 'edgecolor','none')
%                 xlabel('Time (s)')
%                 ylabel('Frequency (Hz)')
%                 view(2)
%                 colorbar
%                 title([type, 'Train zx - Train Error: ', num2str(self.test_error)])
                figure
                surf(self.test_curr', 'edgecolor','none')
                xlabel('Time (s)')
                ylabel('Frequency (Hz)')
                view(2)
                colorbar
                title([type, 'Test zx - Test Error: ', num2str(self.test_error)])   
            else
                self.get_error(self, 0)
                figure
                surf(t, f, self.zx, 'edgecolor','none')
                xlabel('Time (s)')
                ylabel('Frequency (Hz)')
                view(2)
                colorbar
                title('Original')
                figure
                surf(t, f, self.train_curr', 'edgecolor','none')
                xlabel('Time (s)')
                ylabel('Frequency (Hz)')
                view(2)
                colorbar
                title([type, 'Train zx - Train Error: ', num2str(self.test_error)])
                figure
                surf(t, f, self.test_curr', 'edgecolor','none')
                xlabel('Time (s)')
                ylabel('Frequency (Hz)')
                view(2)
                colorbar
                title([type, 'Test zx - Test Error: ', num2str(self.test_error)])            
            end
        end
        
        function plot_evals(self)
            Z = eig(self.OMEGA+self.E*self.BPhi');  %eigenvalues post learning 
            Z2 = eig(self.OMEGA);  %eigenvalues pre learning
            figure()
            plot(Z2,'r.'), hold on 
            plot(Z,'k.') 
            legend('Pre-Learning','Post-Learning')
            xlabel('Re \lambda')
            ylabel('Im \lambda')
        end   
        
        function reset(self, seed)
            rng(seed);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                               PARAMETERS                                %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Learning Parameters  
                self.Pinv = eye(self.N)*self.alpha; %initialize the correlation weight matrix for RLMS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                               SUPERVISOR                                %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                self.imin = round(self.t_start_FORCE/self.dt); % start FORCE after t_start_FORCE seconds
                self.icrit = round(self.t_stop_FORCE/self.dt); % stop FORCE after t_end seconds
                self.nt = round(self.T/self.dt);  % Number of timesteps
                self.k = min(size(self.zx));  % Dimensionality of supervisor
                self.m2 = min(size(self.z2));
                self.nt_HDTS = max(size(self.zx));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                          VARS INITIALISATION                            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Init neurons
                self.v = self.vreset + rand(self.N,1)*(30-self.vreset); % Initialize neuronal voltage with random distribtuions
                self.tlast = zeros(self.N,1); %This vector is used to set  the refractory times 
                self.tspike = zeros(4*self.nt,2); %Storage variable for spike times
            % Init synaptic filters
                self.IPSC = zeros(self.N,1); % post synaptic current storage variable 
                self.h = zeros(self.N,1); %Storage variable for filtered firing rates
                self.r = zeros(self.N,1); %second storage variable for filtered rates 
                self.hr = zeros(self.N,1); % Third variable for filtered rates
                self.JD = 0*self.IPSC; %storage variable required for each spike time 
            % Init weights
                self.E = self.Q*(2*rand(self.N, self.k)-1);  %n
                self.E2 = self.WE2*(2*rand(self.N, self.m2)-1); %HDTS input
                self.OMEGA =  self.G*(randn(self.N,self.N)).*(rand(self.N,self.N)<self.p)/(sqrt(self.N)*self.p); % The initial weight matrix with fixed random weights  
                self.BPhi = zeros(self.N, self.k); % The initial matrix that will be learned by FORCE method
                for i = 1:1:self.N     %set the row average weight to be zero, explicitly. (average input weight to each neuron??)
                    QS = find(abs(self.OMEGA(i,:))>0);
                    self.OMEGA(i, QS) = self.OMEGA(i, QS) - sum(self.OMEGA(i, QS))/length(QS);
                end
                self.kill = ones(size(self.v));
            % Init recording vars 
                self.ns = 0;  % Number of spikes, counts during simulation  
                self.current = zeros(self.nt/10,self.k);  %storage variable for output current/approximant 
                self.orig = zeros(self.nt/10,self.k);  %storage variable for output current/approximant 
                self.z = zeros(self.k, 1);  %Initialize the approximant                 
%                 self.time = 1:1:self.nt;  % Time vector for plotting
                self.RECB = zeros(self.nt/10,10);  % Storage matrix for the synaptic weights (a subset of them) 
        end
    end
end

