%The one that finally worked %Debanjan
clear
%%%%%%%%%%%%Epoching
%[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\newckcyaldageorge\metro1hzleftyalda001\metro1hzleftyaldaS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz

%[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\imageryS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz
[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\newckcyaldageorge\nometro2hzrightyalda001\nometro2hzrightyaldaS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz
signal(1:5120,:)=[]; %for deleting rows

Codes=states.StimulusCode;

Codes(1:5120,:)=[]; %for deleting rows

fs=256;
num_chan=size(signal,2);
sig_norm=(signal);
FreqMin=.5;
FreqMax=100;
Num= fir1(3*fs,[ FreqMin/(fs/2) FreqMax/(fs/2)]);%normalized to Nyquist freq
for cnt=1:num_chan %Number of channels
    signaleeg(:,cnt)=filtfilt(Num,1,double(sig_norm(:,cnt)));
end
 
signaleeg(:,end+1)=Codes(:);
signal3=signaleeg;
 
sample_time=15;
%index3=find(diff(Codes)==4); 
%%%%%% For Tapping
hhhh=signal3(:,end);
num_samples = fs * sample_time;
change1 = hhhh(2:end);
change2 = hhhh(1:end-1);
changes =[0; change1 - change2];
%index3=find(diff(Codes)==4);   
index3=find(changes==-1);
%%%%%%
epoch_data_new_zero_ckc = {zeros(num_samples, num_chan, length(index3))};% creating a zero matrix
cut_T=round(1*fs); %%%%%%%%%%%%%%%% use it
 
for j = 1: length(index3)-1 
    epoch_data_new_zero_ckc{j} = signal3(index3(j):index3(j)+num_samples,1:end-1);
end
  
EEG1= cat(3,epoch_data_new_zero_ckc{:});



%%%%%%%%%%%%%%%%%%
%load sampleEEGdata
%y = reshape(signal, 256, 10);
%V=signal(:,1:13);
V=EEG1;
%V=permute(reshape(V,5121, 13, 17), [2 1 3]);
V=permute(reshape(V,3841, 14, 8), [2 1 3]);

ch2use = 11;
EEG.srate=256;
min_freq =  1;
max_freq = 40;
num_frex = 30;
EEG.times1 = linspace(0,15003,3841); %%Timepoint for data x2+300/datapt-1=spacing,datapt=4684
str={ 'FC3','FC4','C1','Cz','C2','C3','C4','Cp1','Cp2','cp3','Cp4','p3','pz','p4'}; % Vibro
str1 = string(str);

% define wavelet parameters
time = -1:1/EEG.srate:1;
frex = logspace(log10(min_freq),log10(max_freq),num_frex);
s    = logspace(log10(3),log10(13),num_frex)./(2*pi*frex);
% s    =  3./(2*pi*frex); % this line is for figure 13.14
% s    = 10./(2*pi*frex); % this line is for figure 13.14

% definte convolution parameters
EEG.pnts=length(V);
EEG.trials=8;
n_wavelet            = length(time);
n_data               = length(V)*EEG.trials;
n_convolution        = n_wavelet+n_data-1;
n_conv_pow2          = pow2(nextpow2(n_convolution));
half_of_wavelet_size = (n_wavelet-1)/2;

% get FFT of data
eegfft = fft(reshape(V(ch2use,:,:),1,EEG.pnts*EEG.trials),n_conv_pow2);

% initialize
eegpower = zeros(num_frex,EEG.pnts); % frequencies X time X trials

baseidx = dsearchn(EEG.times1',[3000 4000]');

% loop through frequencies and compute synchronization
for fi=1:num_frex
    
    wavelet = fft( sqrt(1/(s(fi)*sqrt(pi))) * exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*(s(fi)^2))) , n_conv_pow2 );
    
    % convolution
    eegconv = ifft(wavelet.*eegfft);
    eegconv = eegconv(1:n_convolution);
    eegconv = eegconv(half_of_wavelet_size+1:end-half_of_wavelet_size);
    
    % Average power over trials (this code performs baseline transform,
    % which you will learn about in chapter 18)
    temppower = mean(abs(reshape(eegconv,EEG.pnts,EEG.trials)).^2,2);
    eegpower(fi,:) = 10*log10(temppower./mean(temppower(baseidx(1):baseidx(2))));
end

figure
%subplot(121)
contourf(EEG.times1,frex,eegpower,40,'linecolor','none')
set(gca,'clim',[-3 3],'xlim',[1000 15000],'yscale','log','ytick',logspace(log10(min_freq),log10(max_freq),6),'yticklabel',round(logspace(log10(min_freq),log10(max_freq),6)*10)/10)
title('Logarithmic frequency scaling')
colormap('jet')
h = colorbar;
ylabel(h, 'dB change from baseline')
title(sprintf('Color limit of -12 to +12 dB %s', str1(ch2use)));
%%%
% subplot(122)
% figure
% contourf(EEG.times1,frex,eegpower,40,'linecolor','none')
% set(gca,'clim',[-3 3],'xlim',[4000 20000])
% %title('Linear frequency scaling')
% colormap('jet')
% h = colorbar;
% ylabel(h, 'dB change from baseline')
% title(sprintf('Linear frequency scaling %s', str1(ch2use)));
% %%%
% 
%%%%%%%%%%%%%%%%%%%%% Time ERP plots


% compute ERP
erp = double(squeeze(mean(V(ch2use,:,:),3)));

% low-pass filter data (requires signal processing toolbox) 
% you'll learn about how filtering works, and what this code means, in chapter 14. 
nyquist          = fs/2;
transition_width = 0.15; % percent


% filter from 5-15
filter_low    =  8; % Hz
filter_high   = 13; % Hz
ffrequencies  = [ 0 filter_low*(1-transition_width) filter_low filter_high filter_high*(1+transition_width) nyquist ]/nyquist;
idealresponse = [ 0 0 1 1 0 0 ];
filterweights = firls(round(3*(fs/filter_low)),ffrequencies,idealresponse);
erp_5to15     = filtfilt(filterweights,1,double(erp));



% now plot all filtered ERPs
figure
plot(EEG.times1,erp_5to15 ,'b','linew',2)

set(gca,'xlim',[3000 10000],'ydir','r')
xlabel('Time (ms)'), ylabel('Voltage (\muV)')
title([ 'ERP from electrode ' ch2use ])
legend({'8-13'})












