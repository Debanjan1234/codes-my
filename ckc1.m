%%%data load
clear
% EEG2=load('p2.mat');
% kk=load('EEg1.mat');
% kkk=kk.EEG1;
% %Flex2=load('flex3.mat');
% %Flex2=Flex2.Flex1;
% %EEG2=EEG2.EEG1;
% p2_avg2=EEG2.p2_avg1;
% %p2_avg2=load('p2avg2'); MetroVanessaL2HzS001R01, EvanRnometro2hzS001R01.dat
% %signal=load('C:\Users\NeuralPC\Desktop\new.txt').';
%%used sawR2hzAct1.csv vs sawR2hzAct1S001R01.dat, sawL1hzAct1.csv vs sawL1hzAct2S001R01.dat
%%sawL1hzAct2S001R01.dat,sawL1hzAct1.csv vs DebanjanR1hzactS001R01.dat,
%%sawR2hznometroact02.csv and SawR1HznoMetroactS002R02.dat;
%sawR2hzmetroact01.csv  vs SawR2HzMetroactS001R01.dat;  EvanR1Hznometro.csv
%EvanR2Hznometro.csv
%EvanRnometro1hzS001R01.dat EvanRnometro2hzS001R01.dat
% EvanR2Hznometro.csv
%filename = 'Glove_Test_L_3_15_18.csv';
M = xlsread('C:\Users\NeuralPC\Desktop\10-26\newdata\newdata1\EvanR2Hznometro.csv');
%M=vertcat(M2,M1);
%signal=vertcat(signal1,signal);
M=detrend(M);
M=abs(M);
%figure
%plot(M)
%xlabel('Sample number')
%ylabel('Rectified Flex signal');
%Metro1HzRightSawyercorrectS001R01 noMetro2HzRightalyssaS001R01.dat
%[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\newckcyaldageorge\nometro2hzleftyalda001\nometro2hzleftyaldaS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz
%[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\imageryS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz

[signal, states, parameters ] = load_bcidat('C:\Users\NeuralPC\Desktop\newckcyaldageorge\metro1hzleftyalda001\metro1hzleftyaldaS001R01.dat'); %% Face paradigm Ch = Fz cZ, P3,Pz,p4p07 p08 oz

Codes=states.StimulusCode;
%Codes=vertcat(states1.StimulusCode,Codes);
%Codes=signal(:,15);
%signal=detrend(signal);
%%%%%Filter%%%
fs=256;
num_chan=size(signal,2);
sig_norm=zscore(signal);
%sig_norm=(signal);
%datatime = linspace(0,214.9,55039);
%Ts = 1/mean(diff(datatime));
%zero phase  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FreqMin=.5;
FreqMax=100;
Num= fir1(3*fs,[ FreqMin/(fs/2) FreqMax/(fs/2)]);%normalized to Nyquist freq
for cnt=1:num_chan %Number of channels
    signaleeg(:,cnt)=filtfilt(Num,1,double(sig_norm(:,cnt)));%no phase delay signal4 zerophase
end
sig_normflex1=zscore(M);
%sig_normflex1=(M);

%sig_normflex=(M);
signalflex=filtfilt(Num,1,sig_normflex1);%no phase delay  zerophase

%%
%%%common average%%%
%clear signal2

       %for cnt=1:num_chan
          % signaleeg(:,cnt)=signaleeg(:,cnt)-mean(signaleeg,2);%Common average referencing signal2
       %end
   

%%%%% Resample
clear signal3 y
signaleeg(:,end+1)=Codes(:);
%signal3 = resample(signaleeg,256,256);
signal3=signaleeg;
sig_normflex = resample(signalflex,256,257);
%sig_normflex = signalflex;

%%upample Flex as labels will be from EEG%%%
hhhh=signal3(:,end);
%hhhh(hhhh>0)=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%
sample_time=15;
num_samples = fs * sample_time;
clear change %%%for p300 Target epoching
change1 = hhhh(2:end);
change2 = hhhh(1:end-1);
changes =[0; change1 - change2];
%index = find(changes); 
%x=[0 0 0 0 1 1 1];
%hhhh=detrend(hhhh);
%index3=find(diff(Codes)==4);  
index2=find(changes==1);
index3=find(changes==-1);
%ff=find(index(2:end)-index(1:end-1)>1000);
clear index2  %% use only in comparison caution
index2=index3;
%I have used cell, as I found it convenient
epoch_data_new_zero_ckc = {zeros(num_samples, num_chan, length(index2))};% creating a zero matrix
cut_T=round(1*fs); %%%%%%%%%%%%%%%% use it

for j = 1: length(index2) -1 %Change remember
    epoch_data_new_zero_ckc{j} = signal3(index2(j)+300:index2(j)+num_samples,:);
end

for j = 1: length(index2) -1
    epoch_data_new_zero_ckc_Flex{j} = sig_normflex(index2(j)+300:index2(j)+num_samples,3); %%choosing only 3rd column
end

EEG1= cat(3,epoch_data_new_zero_ckc{:});
Flex1= cat(2,epoch_data_new_zero_ckc_Flex{:});
%Flex1=detrend(Flex1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5note v=permute(reshape(V, 6400, 14, 1), [3 1 2]); Interesting


%y=signalflex;
%clear flx1 flx2 eeg1 eeg2 Pxx1 Pxx2
%flx1=y(1240:10457,3);
%flx1=y(1199:12522,3);
%flx2=y(13650:15750,3);
%flx1=y(1290:11520,3);

%flx3=y(37100:52480,3);
%eeg1=signal3(1199:12522,:);
%eeg1=signal3(1390:8800,:);

%eeg2=signal3(1290:11520,:);
%eeg3=signal2(37100:52480,:);
%Final=horzcat(Pxx1,Pxx2,Pxx2);
%Finalavg=mean(Final,2);
for i=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
[Pxxx1{i},fz]=mscohere(Flex1(:,1),EEG1(:,i,1),[],[],[],256);%Loo
[Pxxx2{i},fz]=mscohere(Flex1(:,2),EEG1(:,i,2),[],[],[],256);%Loo
[Pxxx3{i},fz]=mscohere(Flex1(:,3),EEG1(:,i,3),[],[],[],256);%Loo
[Pxxx4{i},fz]=mscohere(Flex1(:,4),EEG1(:,i,4),[],[],[],256);%Loo
[Pxxx5{i},fz]=mscohere(Flex1(:,5),EEG1(:,i,5),[],[],[],256);%Loo
[Pxxx6{i},fz]=mscohere(Flex1(:,6),EEG1(:,i,6),[],[],[],256);%Loo
[Pxxx7{i},fz]=mscohere(Flex1(:,7),EEG1(:,i,7),[],[],[],256);%Loo
[Pxxx8{i},fz]=mscohere(Flex1(:,8),EEG1(:,i,8),[],[],[],256);%Loo

end
%[Pxx3,f]=mscohere(flx3,eeg3(:,7),[],[],[],256);%Loo

V1 = cat(2,Pxxx1{:});
V2= cat(2,Pxxx2{:});
V3= cat(2,Pxxx3{:});
V4=cat(2,Pxxx4{:});
V5=cat(2,Pxxx5{:});
V6=cat(2,Pxxx6{:});
V7=cat(2,Pxxx7{:});

Vall= V1+V3;
Vavg=(V1+V2+V3+V4+V5+V6+V7)/7;
figure
subplot(2,2,1)
plot(fz,Vavg(:,1),'LineWidth',1,'color','r'); %%2,8,9 were bad
hold on
plot(fz,Vavg(:,3),'LineWidth',1,'color','b');
hold on
plot(fz,Vavg(:,2),'LineWidth',1,'color','g');
hold on
plot(fz,Vavg(:,7),'LineWidth',1,'color','y');
hold on
plot(fz,Vavg(:,10),'LineWidth',1,'color','c');
hold on
plot(fz,Vavg(:,13),'LineWidth',1,'color','m');
xlim([0 5]);
legend(sprintf('Fc3 '),sprintf('C1'),sprintf('FC4'),sprintf('C4'),sprintf('Cp3'),sprintf('Pz'));
title('Magnitude-Squared Coherence')
xlabel('f (Hz)')
ylim([0 .7])
%%%Topo%%%%
subplot(2,2,2)
%max=find(max(Vavg(:,1)));
topoplot_YS(Vavg(5,1:14),'eloc13sort.txt','EEG')
max=max(Vavg(5,1:14));
min=min(Vavg(5,1:14));
%topoplot_YS1(Vavg(9,1:14),'eloc13sort.txt','EEG','interplimits','electrodes')
caxis([min max ]);
colormap(jet)
colorbar
%colorbar('Direction','reverse')
%caxis('auto');
%max=max(abs(V(9,:)));
%min=min(abs(V(9,:)));

%colorbar
%mscohere(flx1,eeg1(:,8),[],[],[],256);
%hold on
%mscohere(flx1,eeg1(:,10),[],[],[],256);
%hold on
%mscohere(flx1,eeg1(:,13),[],[],[],256);
%hold on
%mscohere(flx1,eeg1(:,3),[],[],[],256);

%xlabel('f (Hz)')
%ylim([0 .6])
%xlim([0 3])
% figure
% subplot(2,2,1)
% hhfft=fft(EEG1(:,3,3)); %% I think get the avg of all the ch
% Fs = 256;            % Sampling frequency                    
% T = 1/Fs;             % Sampling period       
% L = length(EEG1);             % Length of signal
% t = (0:L-1)*T;    % Time vector
% P2 = abs(hhfft/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% P1=abs(P1).^2;
% f = Fs*(0:(L/2))/L;
% plot(f,P1,'DisplayName','noactivity','LineWidth',2);
% xlim([0 10])
% ylim([0 .7])
% %title('Single-Sided Amplitude Spectrum of EEG')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% hold on
% 
% hhfft2=fft(EEG2(:,3,3)); %% I think get the avg of all the ch
% Fs = 256;            % Sampling frequency                    
% T = 1/Fs;             % Sampling period       
% L1 = length(EEG2);             % Length of signal
% t1 = (0:L1-1)*T;    % Time vector
% P3 = abs(hhfft2/L1);
% P4 = P3(1:L1/2+1);
% P4(2:end-1) = 2*P4(2:end-1);
% p4=abs(P4).^2;
% f = Fs*(0:(L/2))/L;
% plot(f,p4,'DisplayName','activity','LineWidth',2,'color','r');
% xlim([0 40])
% ylim([0 .7])
% title('Single-Sided Amplitude Spectrum of EEG')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 
% 
% 
% 
% 
% 
% subplot(2,2,2)
% hhfft1=fft(Flex2(:,3));
% Fs = 256;            % Sampling frequency                    
% T = 1/Fs;             % Sampling period       
% L1 = length(Flex1);             % Length of signal
% t = (0:L-1)*T;    % Time vector
% P3 = abs(hhfft1/L);
% P4 = P3(1:L/2+1);
% P4(2:end-1) = 2*P4(2:end-1);
% f = Fs*(0:(L/2))/L;
% P4=abs(P4).^2;
% plot(f,P4)
% xlim([0 10])
% %ylim([0 .2])
% title('Single-Sided Amplitude Spectrum of Flex')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
%%%%%%%%%%%%%FFT Topoplots%%%%%%
figure
 clear epoch_of_Choice
        epoch_of_Choice=epoch_data_new_zero_ckc;% change this to epoch_data_new_zero1 or epoch_data_new_zero3
        clear  p2_new_power  h{u}
        for u=1:length(epoch_of_Choice)
            L=length( epoch_of_Choice{1, 1});
            h{u}= epoch_of_Choice{3:end}; % Change the epoch legth for proper peak
            p2_new_power{u}=abs(fft(h{u})/L).^2;
        end
        
       
        
        %%%%%%
        
        
        
%             %averaging epochs***FFT
%clear p2_avg1
%p2_avg2=p2_avg1;
     p2_avg1=squeeze(mean(cat(3,p2_new_power{:}),3));%for FFT
          % p2_avg2=squeeze(mean(cat(3,p2_new_power{:}),3));%for FFT

        clear nnnn pppp Pz22 Pz23 
        fs=256;
        str={ 'FC3','FC4','C1','Cz','C2','C3','C4','Cp1','Cp2','cp3','Cp4','p3','pz','p4'}; % Vibro
              % str={
              % 'FC3','FC4','C1','Cz','C2','C3','C4','Cp1','Cp2','Cp3','Cp4','p3','pz','p4'};%%
              % Motor

        str1 = string(str);
        mm=0;
        figure
        for i=1:1:14
%             if i>=5
%                 i=i+1;
%             end
            mm=mm+1;
            nnnn= p2_avg1(:,i);
            pppp=p2_avg2 (:,i); % change according to need
            Pz22 = nnnn(1:L/2+1);%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
            Pz22(2:end-1) = 2*Pz22(2:end-1);%change nnnn with uu or yy or pppp
            % f = 0:fs/L:fs/2;
            f1 = fs*(0:(L/2))/L;
            xlim([1 13]);
            ylim([0 1]);
            subplot(5,3,mm)
            plot(f1,10*(Pz22),'LineWidth',2);%%can use 10*log10
%             hold on
%            Pz23 =  pppp(1:L/2+1);%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
%            Pz23(2:end-1) = 2*Pz23(2:end-1);
%           plot(f1,10*(Pz23),'LineWidth',2,'color', 'r');
%             xlim([8 13])
%           legend(sprintf(' Rest'),sprintf('Tap'));
           title(sprintf(' %s', str1(i)));
            xlabel('f (Hz)')
            ylabel('magnitude')
        end
%%%%%Topoplot%%%%%


 % max=max([max(abs(p2_avg1(19,:))),max(abs(p2_avg1(37,:))),max(abs(p2_avg1(25,:))),max(abs(p2_avg1(40,:)))]);
%min=min([min(abs(p2_avg1(19,:))),min(abs(p2_avg1(37,:))),min(abs(p2_avg1(25,:))),min(abs(p2_avg1(40,:)))]);
figure
topoplot_YS1(p2_avg1(7,1:14),'eloc13sort.txt','EEG','interplimits','electrodes')
max1=max(p2_avg1(7,1:14));
min1=min(p2_avg1(7,1:14));
%topoplot_YS1(Vavg(9,1:14),'eloc13sort.txt','EEG','interplimits','electrodes')
%caxis([min1 max1]);
caxis('auto');
colormap(jet)
%colorbar;
title('1Hz')
%%%%Test
%figure
 %nnnn= p2_avg1(:,3);
            %pppp=p2_avg2 (:,i); % change according to need
           % Pz22 = nnnn(1:L/2+1);%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
           % Pz22(2:end-1) = 2*Pz22(2:end-1);%change nnnn with uu or yy or pppp
            % f = 0:fs/L:fs/2;
           % f1 = fs*(0:(L/2))/L;
           % figure
           % plot(f1,10*(Pz33.Pz22),'LineWidth',2);
           % hold on
           %  plot(f1,10*(Pz22),'LineWidth',2);

           % xlim([0 4]);
%%%%Note Findings
%I could see noise but its very low in no vibration, so hypothesis seems to
%be correct
%%% I like using algo that best suits my hypothesis
% 10 sec rest---10 sec activity---5 sec rest 10 Trials with 12 Runs


%%%%%% Test%%%

% figure
% %subplot(2,2,1)
% figure
% Fs = 256;
% t1 = 0:1/Fs:1-1/Fs;
% %x1 = (Flex1(:,5));
% x1 = (M(1:5000,3));
% N1= length(x1);
% xdft1 = fft(x1);
% xdft1 = xdft1(1:N1/2+1);
% psdx1 = (1/(Fs*N1)) * abs(xdft1).^2;
% psdx1(2:end-1) = 2*psdx1(2:end-1);
% freq1 = 0:Fs/length(x1):Fs/2;
% 
% plot(freq1,((psdx1)),'DisplayName','No Metro','LineWidth',2)
% grid on
% title('Periodogram Using FFT Flex')
% xlabel('Frequency (Hz)')
% ylabel('Power/Frequency)')
%             xlim([0 8])
%             ylim([0 1])
%       hold on
%       
%       
%       Fs = 256;
% %t1 = 0:1/Fs:1-1/Fs;
% x4 = (Flex2(:,5));
% N4= length(x4);
% xdft4 = fft(x4);
% xdft4 = xdft4(1:N4/2+1);
% psdx4 = (1/(Fs*N4)) * abs(xdft4).^2;
% psdx4(2:end-1) = 2*psdx4(2:end-1);
% freq4 = 0:Fs/length(x4):Fs/2;
% 
% plot(freq4,((psdx4)),'DisplayName',' Metro','LineWidth',2)
% grid on
% title('Periodogram  Flex')
% xlabel('Frequency (Hz)')
% ylabel('Power/Frequency)')
%             xlim([0 8])
%             %ylim([0 .5])
%   
%             
%% Power Spectrum%%%%%
figure
%subplot(2,2,2)
Fs = 256;
%t = 0:1/Fs:1-1/Fs;
x1 = (kk.EEG1(:,2,1));
N = length(x1);
xdft = fft(x1);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq1 = 0:Fs/length(x1):Fs/2;
plot(freq1,((psdx)),'DisplayName',' Metro','LineWidth',2)
grid on
title('Periodogram  EEG C')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency')
%ylabel('Power/Frequency (dB/Hz)')
xlim([0 8])
%%%%%

hold on

Fs = 256;
t = 0:1/Fs:1-1/Fs;
x2 = (EEG1(:,2,1));
N2 = length(x2);
xdft2 = fft(x2);
xdft2 = xdft2(1:N2/2+1);
psdx2 = (1/(Fs*N2)) * abs(xdft2).^2;
psdx2(2:end-1) = 2*psdx2(2:end-1);
freq2 = 0:Fs/length(x2):Fs/2;
plot(freq2,((psdx2)),'DisplayName','No Metro','LineWidth',2)
grid on
title('Periodogram  EEG C4')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency')
%ylabel('Power/Frequency (dB/Hz)')
xlim([0 8])

%%%%%%%%%%%%%%%%%Timedomain******
% 
%                          p2_avg7=squeeze(mean(cat(3,epoch_of_Choice{:}),3));%for FFT
%                          
% FreqMin1=1;
% FreqMax1=15;
% Num1= fir1(3*fs,[ FreqMin1/(fs/2) FreqMax1/(fs/2)]);%normalized to Nyquist freq
% for cnt=1:num_chan %Number of channels
%     p2_avg8(:,cnt)=filtfilt(Num1,1,double(p2_avg7(:,cnt)));%no phase delay signal4 zerophase
% end
% 
%   samples =1:L;        % Sample Indices Vector
%         Fs = 256;             % Sampling Frequency (Hz)
%         t = samples*1000/Fs;         % Time Vector (seconds)
%         xlabel('ms')
%         plot(t,p2_avg8(:,11));
% %         hold on
% %         plot(t,p2_avg8(:,2));
%             xlim([1000 12000]);
%%% or do from mike

% 
% chan2plot = 8;
% EEG1=permute(reshape(EEG1,3541, 15, 9), [2 1 3]);
% 
% % compute ERP
% erp = double(squeeze(mean(EEG1(chan2plot,:,:),3)));
% 
% % low-pass filter data (requires signal processing toolbox) 
% % you'll learn about how filtering works, and what this code means, in chapter 14. 
% nyquist          = fs/2;
% transition_width = 0.15; % percent
% 
% % first, filter from 0-40
% filter_cutoff = 40; % Hz
% ffrequencies  = [ 0 filter_cutoff filter_cutoff*(1+transition_width) nyquist ]/nyquist;
% idealresponse = [ 1 1 0 0 ];
% filterweights = firls(100,ffrequencies,idealresponse);
% erp_0to40     = filtfilt(filterweights,1,double(erp));
% 
% 
% 
% filter from 5-15
% filter_low    =  5; % Hz
% filter_high   = 13; % Hz
% ffrequencies  = [ 0 filter_low*(1-transition_width) filter_low filter_high filter_high*(1+transition_width) nyquist ]/nyquist;
% idealresponse = [ 0 0 1 1 0 0 ];
% filterweights = firls(round(3*(fs/filter_low)),ffrequencies,idealresponse);
% erp_5to15     = filtfilt(filterweights,1,double(erp));
% 
% EEG.times1 = linspace(0,138320,3541); %%Timepoint for data x2+300/datapt-1=spacing,datapt=4684
% 
% 
% % now plot all filtered ERPs
% figure
% plot(EEG.times1,erp_5to15 ,'r','linew',2)
% 
% set(gca,'xlim',[1000 12000],'ydir','r')
% xlabel('Time (ms)'), ylabel('Voltage (\muV)')
% title([ 'ERP from electrode ' chan2plot ])
% legend({'5-15'})
% 
% 
% 
% 
% 
% 
% 
% 
