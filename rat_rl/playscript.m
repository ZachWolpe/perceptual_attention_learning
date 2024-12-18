%% Play script for Nelson's rat data
inp = '/Users/bcowley/Benslab/DynOCog/perceptual_attention_learning/rat_rl';

dat = load(fullfile(inp, "Attention_Project_BehavioralData_CowleyLab_fxd.mat"));
dat = dat.data;
tmp = num2cell(cellfun(@str2num, {dat.ratID}));
[dat.ratID] = tmp{:};

%% There's no unique ordering of sessions within rats, so we provide one
dat(1).CondNum = 1;
lag_ssn_num = dat(1).SessionNum;
rat_id = 0;
for i = 2:size(dat, 2)
    % code a condition number field for unique orderings (with SessionNum)
    if dat(i).ratID ~= dat(i - 1).ratID
        dat(i).CondNum = 1;
    else
        if dat(i).SessionNum > lag_ssn_num
            dat(i).CondNum = dat(i - 1).CondNum;
        else
            dat(i).CondNum = dat(i - 1).CondNum + 1;
        end
    end
end

%% Recode the relevant modality
tmp = num2cell(cellfun(@(x) strcmpi(x, 'tone'), {dat.RelevantModality}));
[dat.ReleMode] = tmp{:};

for i = 1:size(dat, 2)
    % code stimuli as numeric
    dat(i).StimCode = single(cellfun(@(x) strcmpi(x, 'go'), dat(i).StimulusList));
    % Handle missing values, and code responses as numeric
    idx = cellfun(@isempty, dat(i).Response);
    dat(i).RespCode = single(~idx);
    rsp = dat(i).Response;
    rsp(idx) = {'-'};
    idx = cell2mat(cellfun(@(x) startsWith(x, 'cor', 'Ig', 1), rsp, 'Un', 0));
    dat(i).RespCode(idx) = 1;
    idx = cell2mat(cellfun(@(x) startsWith(x, 'hit', 'Ig', 1), rsp, 'Un', 0));
    dat(i).RespCode(idx) = 2;
    idx = cell2mat(cellfun(@(x) startsWith(x, 'fal', 'Ig', 1), rsp, 'Un', 0));
    dat(i).RespCode(idx) = 3;
    idx = cell2mat(cellfun(@(x) startsWith(x, 'omi', 'Ig', 1), rsp, 'Un', 0));
    dat(i).RespCode(idx) = 4;
    % debug
    testi = unique(rsp);
    if any(~ismember(testi, {'correct_rejection' 'false_alarm' 'hit' 'omission' '-'}))
        disp(dat(i).ratID)
        disp(testi)
    end
end

%% Trim the fat
dat = rmfield(dat, {'StimulusList', 'Response', 'RelevantModality'});

% Make some docs
info = {'SessionType', {'EDS' 'EDS_BL_Easy' 'EDS_BL_Hard' 'IDS' 'IDS_BL_Easy' 'RE' 'Stair'}, {'Acronyms for each session type that occured'};
 'SessionNum', [1 2 3 4 5 6 7 8 9 10 11 12 13], {'Consecutive number of each session within each session type'};
 'CondNum', [1 2 3 4 5 6 7 8 9 10 11 12 13 14], {'Numeric ordering of occurence of conditions - same as numeric code for SessionType IFF these occured in same order for every rat.'};
 'ReleMode', [0 1], {'Logical coding for relevant modality: 0=visual, 1=tone'};
 'StimCode', [0 1], {'Logical coding for stimulus class: 0=nogo, 1=go'};
 'RespCode', [1 2 3 4], {'Numeric coding for response type: 1=correct_rejection, 2=hit, 3=false_alarm, 4=omissions'}};

% Export the dataframe to something non-matlabby?
% tmp = struct2table(dat);
% writestruct(dat, fullfile(inp, "attention_behaviorals.csv"))
save(fullfile(inp, "attention_behaviorals.mat"), "dat", "info")