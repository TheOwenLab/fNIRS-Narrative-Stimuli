%% ISC 

% load in data
filename = "PreprocessedData.mat";
data = load(filename);
data = data.data;
% set loop parameters
nSub = length(data); nCond = 5; 
nChannel = 121; nHb = 2;

% put data in easy to use format
Rest = cat(4,data.Rest); BYD = cat(4,data.BYD); BYDN = cat(4,data.BYDN);
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);
Group = {BYD;BYDN;Taken;TakenNoise;Rest};

% preallocate ISC results
ISC = nan(nCond,nChannel,nHb,nSub);
% for each subject
for ss=1:nSub
    % split subject and mean group
    gs = setdiff(1:nSub,ss);
    % get subject and subject - 1 data for each condition
    for nc=1:nCond
        scdata = Group{nc}(:,:,:,ss); gcdata = Group{nc}(:,:,:,gs);
        % for each channel
        for cc = 1:nChannel
            % finally, for each Hb type
            for hb = 1:nHb
                % fisher transformed version of the correlation between group 
                ISC(nc,cc,hb,ss) = atanh(corr(scdata(:,cc,hb),mean(gcdata(:,cc,hb,:),4)));               
            end
        end
                
    end
    
end
    
save("ISC.mat",'ISC')


