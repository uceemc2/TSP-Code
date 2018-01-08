%******************************************************************************************************************
% Please refer to IEEE TRANSACTIONS ON SIGNAL PROCESSING: 
% " Compressive Sensing with Side Information: How to Optimally Capture this
% Extra Information for GMM Signals?"
% version = 1; for operating experiments with random/designed CASSI measurements (1 snapshot): section - V-B
% version = 2; for operating experiments with random/designed CASSI measurements (2 snapshots): section - V-B
% version = 3; for operating experiments with dictionary learning: section - V-B
%******************************************************************************************************************
function TspCASSI(version)
switch version
    case 1, MainRecCASSISanpshot1()
    case 2, MainRecCASSISanpshot2() 
    case 3, Main_CDL_MeasDesign()
    otherwise, return;
end