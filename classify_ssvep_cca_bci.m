
clc
clear
close all
load('D:\New folder\S2.mat');
load('D:\New folder\Freq_Phase.mat');
%% Initialize parameters
Fs=250; % sampling rate
t_length=5.5; % data length (5.5s)
n_run=6;      % number of runs
targets = 1:2:7;
frequencies=freqs(targets);           % stimulus frequencies 8,10,12,14 Hz
N_freq=length(frequencies);                     % number of stimulus frequencies
%best channels according to an article   %P7=44 P3=46 Pz=48 P4=50 P8=52 O1=61 Oz=62 O2=63
channels = [44,46,48,50,52,61,62,63];
data_ch=data(channels,0.5*Fs+1:end,:,:);
%% CCA 
N=2; 
sig_ref1=stim_sig(frequencies(1),Fs,t_length*Fs,N);
sig_ref2=stim_sig(frequencies(2),Fs,t_length*Fs,N);
sig_ref3=stim_sig(frequencies(3),Fs,t_length*Fs,N);
sig_ref4=stim_sig(frequencies(4),Fs,t_length*Fs,N);
% Recognition
[b,a] = butter(3,1/(Fs/2),'high');
lable=[];
for run=1:6
    for j=1:4
        [wx1,wy1,r1]=cca(filtfilt(b,a,data_ch(:,:,targets(j),run)')',sig_ref1);
        [wx2,wy2,r2]=cca(filtfilt(b,a,data_ch(:,:,targets(j),run)')',sig_ref2);
        [wx3,wy3,r3]=cca(filtfilt(b,a,data_ch(:,:,targets(j),run)')',sig_ref3);
        [wx4,wy4,r4]=cca(filtfilt(b,a,data_ch(:,:,targets(j),run)')',sig_ref4);
        feat_max(:,j)=([max(r1);max(r2);max(r3);max(r4)]);
        lable=[lable,j];
    end
    feaures_cca(:,(run-1)*4+1:run*4)=  feat_max;
end

K = 6;
inds = crossvalind('Kfold',lable,K);
C=0;
for i_fold = 1:K
    test = inds==i_fold;
    train = ~test;
    Model=svm.train(feaures_cca(:,train),lable(train)');
    class=svm.predict(Model,feaures_cca(:,test));
    C= C+confusionmat(class,lable(test));
    %class = multisvm(feaures_cca(:,train),lable(train)',feaures_cca(:,test));
    %acc(i_fold) = 100*length(find(class == lable(test)'))/length(class);
end

accuracy = sum(diag(C)) / sum(C(:)) *100;
accuracy1 = C(1,1) / sum(C(1,:)) *100;
accuracy2 = C(2,2) / sum(C(2,:)) *100;
accuracy3 = C(3,3) / sum(C(3,:)) *100;
accuracy4 = C(4,4) / sum(C(4,:)) *100;
disp(['Total:',num2str(accuracy),' %'])
disp(['f1:   ',num2str(accuracy1),' %'])
disp(['f2:   ',num2str(accuracy2),' %'])
disp(['f3:   ',num2str(accuracy3),' %'])
disp(['f4:   ',num2str(accuracy4),' %'])

