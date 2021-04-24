AHIthr = 15;
aux = sum(AHI<AHIthr);
etiq = [-ones(aux,1);ones(n_studies_30-aux,1)]; % "-1"-> sano, "+1"-> enfermo
labels = [];
for i = 1:n_studies_30
    if AHI(i)>AHIthr
    labels{i} = 'enfermo';
    else
        labels{i} = 'sano';
    end
end

%auc_full_cd = aucscore(etiq,AHIest_TARG',0)

% [X,Y,T,AUC] = perfcurve(labels,AHIout.AHIest_050,'enfermo');
% area under the ROC curve for different thresholds of the MLP neuron output
thresholds = 0.2:0.01:0.95;
auc_full_cd(1) = aucscore(etiq,AHIout.AHIest_020',0);
auc_full_cd(2) = aucscore(etiq,AHIout.AHIest_021',0);
auc_full_cd(3) = aucscore(etiq,AHIout.AHIest_022',0);
auc_full_cd(4) = aucscore(etiq,AHIout.AHIest_023',0);
auc_full_cd(5) = aucscore(etiq,AHIout.AHIest_024',0);
auc_full_cd(6) = aucscore(etiq,AHIout.AHIest_025',0);
auc_full_cd(7) = aucscore(etiq,AHIout.AHIest_026',0);
auc_full_cd(8) = aucscore(etiq,AHIout.AHIest_027',0);
auc_full_cd(9) = aucscore(etiq,AHIout.AHIest_028',0);
auc_full_cd(10) = aucscore(etiq,AHIout.AHIest_029',0);
auc_full_cd(11) = aucscore(etiq,AHIout.AHIest_030',0);
auc_full_cd(12) = aucscore(etiq,AHIout.AHIest_031',0);
auc_full_cd(13) = aucscore(etiq,AHIout.AHIest_032',0);
auc_full_cd(14) = aucscore(etiq,AHIout.AHIest_033',0);
auc_full_cd(15) = aucscore(etiq,AHIout.AHIest_034',0);
auc_full_cd(16) = aucscore(etiq,AHIout.AHIest_035',0);
auc_full_cd(17) = aucscore(etiq,AHIout.AHIest_036',0);
auc_full_cd(18) = aucscore(etiq,AHIout.AHIest_037',0);
auc_full_cd(19) = aucscore(etiq,AHIout.AHIest_038',0);
auc_full_cd(20) = aucscore(etiq,AHIout.AHIest_039',0);
auc_full_cd(21) = aucscore(etiq,AHIout.AHIest_040',0);
auc_full_cd(22) = aucscore(etiq,AHIout.AHIest_041',0);
auc_full_cd(23) = aucscore(etiq,AHIout.AHIest_042',0);
auc_full_cd(24) = aucscore(etiq,AHIout.AHIest_043',0);
auc_full_cd(25) = aucscore(etiq,AHIout.AHIest_044',0);
auc_full_cd(26) = aucscore(etiq,AHIout.AHIest_045',0);
auc_full_cd(27) = aucscore(etiq,AHIout.AHIest_046',0);
auc_full_cd(28) = aucscore(etiq,AHIout.AHIest_047',0);
auc_full_cd(29) = aucscore(etiq,AHIout.AHIest_048',0);
auc_full_cd(30) = aucscore(etiq,AHIout.AHIest_049',0);
auc_full_cd(31) = aucscore(etiq,AHIout.AHIest_050',0);
auc_full_cd(32) = aucscore(etiq,AHIout.AHIest_051',0);
auc_full_cd(33) = aucscore(etiq,AHIout.AHIest_052',0);
auc_full_cd(34) = aucscore(etiq,AHIout.AHIest_053',0);
auc_full_cd(35) = aucscore(etiq,AHIout.AHIest_054',0);
auc_full_cd(36) = aucscore(etiq,AHIout.AHIest_055',0);
auc_full_cd(37) = aucscore(etiq,AHIout.AHIest_056',0);
auc_full_cd(38) = aucscore(etiq,AHIout.AHIest_057',0);
auc_full_cd(39) = aucscore(etiq,AHIout.AHIest_058',0);
auc_full_cd(40) = aucscore(etiq,AHIout.AHIest_059',0);
auc_full_cd(41) = aucscore(etiq,AHIout.AHIest_060',0);
auc_full_cd(42) = aucscore(etiq,AHIout.AHIest_061',0);
auc_full_cd(43) = aucscore(etiq,AHIout.AHIest_062',0);
auc_full_cd(44) = aucscore(etiq,AHIout.AHIest_063',0);
auc_full_cd(45) = aucscore(etiq,AHIout.AHIest_064',0);
auc_full_cd(46) = aucscore(etiq,AHIout.AHIest_065',0);
auc_full_cd(47) = aucscore(etiq,AHIout.AHIest_066',0);
auc_full_cd(48) = aucscore(etiq,AHIout.AHIest_067',0);
auc_full_cd(49) = aucscore(etiq,AHIout.AHIest_068',0);
auc_full_cd(50) = aucscore(etiq,AHIout.AHIest_069',0);
auc_full_cd(51) = aucscore(etiq,AHIout.AHIest_070',0);
auc_full_cd(52) = aucscore(etiq,AHIout.AHIest_071',0);
auc_full_cd(53) = aucscore(etiq,AHIout.AHIest_072',0);
auc_full_cd(54) = aucscore(etiq,AHIout.AHIest_073',0);
auc_full_cd(55) = aucscore(etiq,AHIout.AHIest_074',0);
auc_full_cd(56) = aucscore(etiq,AHIout.AHIest_075',0);
auc_full_cd(57) = aucscore(etiq,AHIout.AHIest_076',0);
auc_full_cd(58) = aucscore(etiq,AHIout.AHIest_077',0);
auc_full_cd(59) = aucscore(etiq,AHIout.AHIest_078',0);
auc_full_cd(60) = aucscore(etiq,AHIout.AHIest_079',0);
auc_full_cd(61) = aucscore(etiq,AHIout.AHIest_080',0);
auc_full_cd(62) = aucscore(etiq,AHIout.AHIest_081',0);
auc_full_cd(63) = aucscore(etiq,AHIout.AHIest_082',0);
auc_full_cd(64) = aucscore(etiq,AHIout.AHIest_083',0);
auc_full_cd(65) = aucscore(etiq,AHIout.AHIest_084',0);
auc_full_cd(66) = aucscore(etiq,AHIout.AHIest_085',0);
auc_full_cd(67) = aucscore(etiq,AHIout.AHIest_086',0);
auc_full_cd(68) = aucscore(etiq,AHIout.AHIest_087',0);
auc_full_cd(69) = aucscore(etiq,AHIout.AHIest_088',0);
auc_full_cd(70) = aucscore(etiq,AHIout.AHIest_089',0);
auc_full_cd(71) = aucscore(etiq,AHIout.AHIest_090',0);
auc_full_cd(72) = aucscore(etiq,AHIout.AHIest_091',0);
auc_full_cd(73) = aucscore(etiq,AHIout.AHIest_092',0);
auc_full_cd(74) = aucscore(etiq,AHIout.AHIest_093',0);
auc_full_cd(75) = aucscore(etiq,AHIout.AHIest_094',0);
auc_full_cd(76) = aucscore(etiq,AHIout.AHIest_095',0);

[auc_max,thr_opt] = max(auc_full_cd);
fprintf('El valor máximo de AUC para el método FULL-CD es: %.4f \n',auc_max)
fprintf('El umbral óptimo para el método FULL-CD es: %.2f \n',thresholds(thr_opt))

% plots

plot(thresholds,auc_full_cd,'k',thresholds(thr_opt),auc_max,'k*')
axis([0.2 0.95 0 1])
grid on
xlabel('thresholds')
ylabel('AUC value')
hold on

