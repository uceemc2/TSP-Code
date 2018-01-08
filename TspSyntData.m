%*******************
% Please refer to IEEE TRANSACTIONS ON SIGNAL PROCESSING: 
% " Compressive Sensing with Side Information: How to Optimally Capture this
% Extra Information for GMM Signals?"
% version = 1; for operating synthetic data for Gaussian sources (IV-A);
% version = 2; for operating synthetic data for GMM sources (IV-B);
% version = 3; for operating real imaging applications (IV-C);
%*******************
function TspSyntData(version)
switch version
    case 1, Synthetic_GSVD_Gaussian()
    case 2, Synthetic_GSVD_GMM()
    case 3, RealData()
    otherwise, return;
end