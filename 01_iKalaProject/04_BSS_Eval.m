clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Parmaters Setting
ToolDirStr = '../00_Tools/';
WavDirStr = '../Wavfile/';
CNNAudioDirStr = './Audio/Threshold_0_35/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Addpath for SineModel/UtilFunc/BSS_Eval
addpath(genpath(ToolDirStr));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0 - Obtain Audio File Name
WavFileNames = iKalaWavFileNames(WavDirStr);
[VoiceFileNames,SongFileNames] = iKalaCNNFileNames(CNNAudioDirStr);
numMusics = numel(WavFileNames);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BSS = zeros(numMusics,6);
for t = 1:numMusics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 1 - BSS Evaluation
    tic
    [x, fs] = audioread(WavFileNames{t});
    trueVoice = gpuArray(x(:,2));
    trueKaraoke = gpuArray(x(:,1));
    trueMixed = gpuArray(x(:,1)+x(:,2));
    
    Voice.y = audioread(VoiceFileNames{t});
    Song.y = audioread(SongFileNames{t});
    
    estimatedVoice = gpuArray(Voice.y);
    estimatedKaraoke = gpuArray(Song.y);
    [SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / norm(estimatedVoice + estimatedKaraoke), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    [NSDR, ~, ~] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));
    NSDR = SDR - NSDR;
    
    BSS(t,1) = gather(NSDR(1));
    BSS(t,2) = gather(NSDR(2));
    BSS(t,3) = gather(SIR(1));
    BSS(t,4) = gather(SIR(2));
    BSS(t,5) = gather(SAR(1));
    BSS(t,6) = gather(SAR(2));
    
    fprintf('NSDR:%.4f, %.4f\n', NSDR(1), NSDR(2));
    fprintf('SIR:%.4f, %.4f\n', SIR(1), SIR(2));
    fprintf('SAR:%.4f, %.4f\n', SAR(1), SAR(2));
    fprintf('Computing %d BSSEval - (Voice, Song)] - needs %.2f sec\n', t, toc);
end
