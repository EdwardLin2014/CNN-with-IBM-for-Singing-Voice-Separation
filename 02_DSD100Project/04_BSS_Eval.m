% This script is intended to perform the evaluation of the separation
% quality on the Demixing Secrets Dataset 100 (DSD100).
% It is primarily intended for the task of "professionally-produced
% music recordings (MUS)" of community-based Signal Separation
% Evaluation Campaign (SiSEC) (https://sisec.inria.fr/).
%
% This function should be used along with the DSD100.
% * The variable base_estimates_directory stand for the root folder in
% which the script should find subdirectories containing the results of the
% methods you want to evaluate. each of these subdirectories must contain
% the exact same file structure than the DSD dataset, as produced by the
% DSD100_separate_and_eval_parallel.m script or the python dsd100 package.
% The matching is case sensitive. There is the possibility of not including
% all sources, and also to include the "accompaniment" source, defined as
% the sum of all sources except vocals.
%
% * the variable dataset_folder links to the root folder of the DSD100
% dataset.
%
% The evaluation function should then be called simply as follows:
%   DSD100_only_eval.m
% The function loops over all the 100 songs of the DSD100 data set, and,
% for each song, for both the "Dev" and "Test" subsets, performs evaluation
% using the BSS Eval toolbox 3.0 (included in this function) and saves the
% results (i.e. SDR, ISR, SIR, and SAR) in the file "results.mat," including
% the song name. The function also saves the results for all the songs in a
% single file "resultX.mat" to the root folder, along with this function.
%
% A first evaluation is performed for the 4 sources vocals/bass/drums
% and other, and a second is performed for accompaniment.
%
% We would like to thank Emmanuel Vincent for giving us the permission to
% use the BSS Eval toolbox 3.0;
%
% If you use this script, please reference the following paper
%
%@inproceedings{SiSEC2015,
%  TITLE = {{The 2015 Signal Separation Evaluation Campaign}},
%  AUTHOR = {N. Ono and Z. Rafii and D. Kitamura and N. Ito and A. Liutkus},
%  BOOKTITLE = {{International Conference on Latent Variable Analysis and Signal Separation  (LVA/ICA)}},
%  ADDRESS = {Liberec, France},
%  SERIES = {Latent Variable Analysis and Signal Separation},
%  VOLUME = {9237},
%  PAGES = {387-395},
%  YEAR = {2015},
%  MONTH = Aug,
%}
%
% A more adequate reference will soon be given at sisec.inria.fr
% please check there and provide the one provided.
%
% Original Author: Zafar Rafii
% Updated by A. Liutkus
% Further updated by Kin Wah Edward LIN

clear all; close all; clc

%here provide the name of the directory that contains all results from your
%different techniques as subfolders
base_estimates_directory = './Audio';

%here, provide the root folder for the DSD100 dataset.
dataset_folder = './Wavfile';


warning('off','all')
subsets_names = {'Dev','Test'};
sources_names = {'bass','drums','other','vocals','accompaniment'};


methods_folders = dir(base_estimates_directory);
isub = [methods_folders(:).isdir]; %# returns logical vector
methods_folders = {methods_folders(isub).name}';
methods_folders(ismember(methods_folders,{'.','..'})) = [];

%comment this line if you don't want to use parallel toolbox. This may
%happen for memory usage reasons. Also, modify this line if you don't want
%to use the maximum number of cores, which is the default behaviour.
%parpool('local');

result = struct;
results_set=cell(50,1);
%for imethod_name = 1:length(methods_folders)
for imethod_name = 1
    %for each method (subfolder of the results root folder provided)
    method_name = methods_folders{imethod_name};
    fprintf('\n computing results for method %s\n', method_name)
    
    %loop over dev and test subsets
    for subset_index = 1:2
        sources_folder = fullfile(dataset_folder,'Sources',subsets_names{subset_index});
        estimates_folder = fullfile(base_estimates_directory,method_name,subsets_names{subset_index});
        files_list = dir(estimates_folder);
        numFiles = numel(files_list);
        
        %loop over songs
        song_index = 1;
        for file_index = 1:numFiles
            song_name = files_list(file_index).name;
            if strcmp(song_name,'.') || strcmp(song_name,'..') || strcmp(song_name,'.DS_Store')
                continue;
            end
            disp([subsets_names{subset_index},' ',num2str(song_index),'/',num2str(50),' ',song_name])
            
            %load the sources references
            sources_data = [];
            for source_index = 1:4
                source_file = fullfile(sources_folder,song_name,[sources_names{source_index},'.wav']);
                [source_data,source_sampling] = audioread(source_file);
                [sources_samples,sources_channels] = size(source_data);
                source_data = repmat(source_data,[1,3-sources_channels]);
                sources_data = cat(3,sources_data,source_data);
            end
            %build accompaniment reference as sum of sources
            accompaniment_data = sum(sources_data(:,:,1:3),3);
            
            %get estimated sources
            sources_estimates_data = zeros(sources_samples,2,4);
            for estimate_index = 1:4
                estimate_file = fullfile(estimates_folder,song_name,[sources_names{estimate_index},'.wav']);
                if exist(estimate_file,'file') == 2
                    estimate_data = audioread(estimate_file);
                    [estimate_samples,estimate_channels] = size(estimate_data);
                    %if mono estimate: duplicate it to get stereo
                    estimate_data = repmat(estimate_data,[1,3-estimate_channels]);
                    
                    estimates_samples = min(size(sources_estimates_data,1),estimate_samples);
                    sources_estimates_data = sources_estimates_data(1:estimates_samples,:,:);
                    sources_estimates_data(:,:,estimate_index) = estimate_data(1:estimates_samples,:);
                end
            end
            accompaniment_estimate_file = fullfile(estimates_folder,song_name,[sources_names{5},'.wav']);
            if exist(accompaniment_estimate_file,'file') == 2
                %If there's an accompaniment estimated file, load it.
                estimate_data = audioread(accompaniment_estimate_file);
                [estimate_samples,estimate_channels] = size(estimate_data);
                estimate_data = repmat(estimate_data,[1,3-estimate_channels]);
                estimates_samples = min(size(sources_estimates_data,1),estimate_samples);
                sources_estimates_data = sources_estimates_data(1:estimates_samples,:,:);
                accompaniment_estimates_data = estimate_data(1:estimates_samples,:);
            else
                %else: sum sources estimates except vocals
                accompaniment_estimates_data = sum(sources_estimates_data(:,:,1:3),3);
            end
            tic
            %estimate quality for accompaniment
            [SDRac,ISRac,SIRac,SARac] = bss_eval( gpuArray(cat(3,sources_estimates_data(:,:,4),accompaniment_estimates_data)),...
                gpuArray(cat(3,sources_data(:,:,4),accompaniment_data)),...
                30*source_sampling,15*source_sampling);
            
            %estimate quality for sources
            [SDR,ISR,SIR,SAR] = bss_eval(gpuArray(sources_estimates_data),gpuArray(sources_data),...
                30*source_sampling,15*source_sampling);
            toc
            
            %append the quality of accompaniment to the results for sources
            SDR(end+1,:) = SDRac(2,:);
            ISR(end+1,:) = ISRac(2,:);
            SIR(end+1,:) = SIRac(2,:);
            SAR(end+1,:) = SARac(2,:);
            
            %build the result structure for this song
            results_set{song_index} = struct;
            results_set{song_index}.name = song_name;
            for source_index = 1:5
                results_set{song_index}.(sources_names{source_index}).sdr = gather(SDR(source_index,:));
                results_set{song_index}.(sources_names{source_index}).isr = gather(ISR(source_index,:));
                results_set{song_index}.(sources_names{source_index}).sir = gather(SIR(source_index,:));
                results_set{song_index}.(sources_names{source_index}).sar = gather(SAR(source_index,:));
            end
            song_index = song_index + 1;
        end
        % now gather all the results for the subset
        song_index = 1;
        for file_index=1:numFiles
            song_name = files_list(file_index).name;
            if strcmp(song_name,'.') || strcmp(song_name,'..') || strcmp(song_name,'.DS_Store')
                continue;
            end
            results_file = fullfile(estimates_folder,song_name,'results.mat');
            results = results_set{song_index};
            save(results_file,'results')
            result.(lower(subsets_names{subset_index}))(song_index).results =results_set{song_index};
            song_index = song_index + 1;
        end
    end
    %save the results for the method
    result_file = fullfile(base_estimates_directory,[method_name,'.mat']);
    save(result_file,'result')
    warning('on','all')
end

%Now, bsseval code
function [SDR,ISR,SIR,SAR,Gj,G] = bss_eval(ie,i,win,ove,Gj,G)
    [nsampl,~,nsrc] = size(ie);
    nwin = floor((nsampl-win+1+ove)/ove);
    SDR = gpuArray(zeros(nsrc,nwin));
    ISR = gpuArray(zeros(nsrc,nwin));
    SIR = gpuArray(zeros(nsrc,nwin));
    SAR = gpuArray(zeros(nsrc,nwin));
    if (nargin == 4) || isempty(Gj) || isempty(G)
        Gj = cell(nsrc,nwin);
        G = cell(nwin,1);
    end
    for k = 1:nwin
        K = (k-1)*ove+1:(k-1)*ove+win;
        [SDR(:,k),ISR(:,k),SIR(:,k),SAR(:,k),Gj_k,G_k] = bss_eval_images(ie(K,:,:),i(K,:,:),Gj(:,k),G{k});
        Gj(:,k)=Gj_k;
        G{k}=G_k;
    end
end

function [SDR,ISR,SIR,SAR,Gj,G] = bss_eval_images(ie,i,Gj,G)
    nsrc = size(ie,3);
    if nargin == 2
        %if the G matrix is not given, initialize it to empty
        Gj = cell(nsrc,1);
        G = [];
    end
    SDR = gpuArray(zeros(nsrc,1));
    ISR = gpuArray(zeros(nsrc,1));
    SIR = gpuArray(zeros(nsrc,1));
    SAR = gpuArray(zeros(nsrc,1));
    for j = 1:nsrc
        [s_true,e_spat,e_interf,e_artif,Gj_temp,G] = bss_decomp_mtifilt(ie(:,:,j),i,j,512,Gj{j},G);
        Gj{j}=Gj_temp;
        [SDR(j,1),ISR(j,1),SIR(j,1),SAR(j,1)] = bss_image_crit(s_true,e_spat,e_interf,e_artif);
    end
end

function [s_true,e_spat,e_interf,e_artif,Gj,G] = bss_decomp_mtifilt(se,s,j,flen,Gj,G)

    nchan = size(se,2);
    s_true = [s(:,:,j);zeros(flen-1,nchan)];
    [e_spat,Gj] = project(se,s(:,:,j),flen,Gj);
    e_spat=e_spat-s_true;
    [e_interf,G] = project(se,s,flen,G);
    e_interf=e_interf-s_true-e_spat;
    e_artif = [se;zeros(flen-1,nchan)]-s_true-e_spat-e_interf;
end

function [sproj,G] = project(se,s,flen,G)
    warning('off','all')
    [nsampl,nchan,nsrc] = size(s);
    s = reshape(s,[nsampl,nchan*nsrc]);
    s = [s;zeros(flen-1,nchan*nsrc)];
    se = [se;zeros(flen-1,nchan)];
    fftlen = 2^nextpow2(nsampl+flen-1);
    sf = fft(s',fftlen,2);
    sef = fft(se',fftlen,2);
    if isempty(G)
        G = gpuArray(zeros(nchan*nsrc*flen));
        for k1 = 0:nchan*nsrc-1
            for k2 = 0:k1
                ssf = sf(k1+1,:).*conj(sf(k2+1,:));
                ssf = real(ifft(ssf));
                ss = toeplitz(ssf([1,fftlen:-1:fftlen-flen+2]),ssf(1:flen));
                G(k1*flen+1:k1*flen+flen,k2*flen+1:k2*flen+flen) = ss;
                G(k2*flen+1:k2*flen+flen,k1*flen+1:k1*flen+flen) = ss';
            end
        end
    end
    D = gpuArray(zeros(nchan*nsrc*flen,nchan));
    for k = 0:nchan*nsrc-1
        for i = 1:nchan
            ssef = sf(k+1,:).*conj(sef(i,:));
            ssef = real(ifft(ssef,[],2));
            D(k*flen+1:k*flen+flen,i) = ssef(:,[1,fftlen:-1:fftlen-flen+2])';
        end
    end

    C = G\D;
    C = reshape(C,flen,nchan*nsrc,nchan);
    sproj = gpuArray(zeros(nsampl+flen-1,nchan));
    for k = 1:nchan*nsrc
        for i = 1:nchan
            sproj(:,i) = sproj(:,i)+fftfilt(C(:,k,i),s(:,k));
        end
    end
end

function [SDR,ISR,SIR,SAR] = bss_image_crit(s_true,e_spat,e_interf,e_artif)
    s_true = s_true(:);
    e_spat = e_spat(:);
    e_interf = e_interf(:);
    e_artif = e_artif(:);
    SDR = 10*log10(sum(s_true.^2)/sum((e_spat+e_interf+e_artif).^2));
    ISR = 10*log10(sum(s_true.^2)/sum(e_spat.^2));
    SIR = 10*log10(sum((s_true+e_spat).^2)/sum(e_interf.^2));
    SAR = 10*log10(sum((s_true+e_spat+e_interf).^2)/sum(e_artif.^2));
end