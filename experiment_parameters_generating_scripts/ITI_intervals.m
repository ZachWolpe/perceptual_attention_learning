NumTotalTrials = 60;
NumIntervals = 10;

interval_array = [];

for i = 1:200
    disp('i = ')
    disp(i)
    intervals = GenerateFlatHazardFunction(NumIntervals, NumTotalTrials)
    interval_array = [interval_array; intervals]
end

csvwrite('intervals.csv', interval_array)


function [ITI_list] = GenerateFlatHazardFunction(NumIntervals, NumTotalTrials)
x = 1:NumIntervals;
y = [];
y = geocdf(x,1/(NumIntervals/2)); %Geometric cumulative distribution function
y = horzcat(0,y); % add 0 to edge
selDist = rand(1,NumTotalTrials); % Select random randomly (with replacement) from distribution
[~,binMembership] = histc(selDist,y); % If bin membership =0, then it can be catch trial (no stimulus) to see false alarm rate
clear y selDist x
ITI_list = binMembership;
clear binMembership
end
