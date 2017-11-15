function [ VoiceFileNames, SongFileNames ] = iKalaCNNFileNames(DatabaseDirStr)
%% Output 
% FileDirs = cell(252,1);
% 137 Verse = cell(1:137,1);
% 115 Chorus = cell(138:252,1);

%% Function Body
VoiceDirStr = [DatabaseDirStr, 'Voice/'];
CNN = dir(VoiceDirStr);
numFiles = 0;
startIdx = 0;
for i = 1:numel(CNN)
    filename = CNN(i).name;
    if numel(filename) > 3
        if strcmp(filename(end-3:end), '.wav')
            numFiles = numFiles + 1;
        else
            startIdx = i;
        end
    else
        startIdx = i;
    end
end

VoiceFileNames = cell(numFiles,1);
l = 0;
for i = startIdx+1:numFiles+startIdx
    wavname = CNN(i).name;
    if strcmp(wavname(end-9:end), '_verse.wav')
        l = l + 1;
        VoiceFileNames{l} = [VoiceDirStr, wavname];
    end
end
for i = startIdx+1:numFiles+startIdx
    wavname = CNN(i).name;
    if strcmp(wavname(end-10:end), '_chorus.wav')
        l = l + 1;
        VoiceFileNames{l} = [VoiceDirStr, wavname];
    end
end

SongDirStr = [DatabaseDirStr, 'Song/'];
CNN = dir(SongDirStr);
numFiles = 0;
startIdx = 0;
for i = 1:numel(CNN)
    filename = CNN(i).name;
    if numel(filename) > 3
        if strcmp(filename(end-3:end), '.wav')
            numFiles = numFiles + 1;
        else
            startIdx = i;
        end
    else
        startIdx = i;
    end
end

SongFileNames = cell(numFiles,1);
l = 0;
for i = startIdx+1:numFiles+startIdx
    wavname = CNN(i).name;
    if strcmp(wavname(end-9:end), '_verse.wav')
        l = l + 1;
        SongFileNames{l} = [SongDirStr, wavname];
    end
end
for i = startIdx+1:numFiles+startIdx
    wavname = CNN(i).name;
    if strcmp(wavname(end-10:end), '_chorus.wav')
        l = l + 1;
        SongFileNames{l} = [SongDirStr, wavname];
    end
end

end

