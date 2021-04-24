% obtain performance measures for all of methods

disp('--------------------------------------------------')
disp(['Method: ' metodo])
disp('--------------------------------------------------')

% thresholds and targets settings
disp('--------------------------------------------------')
disp('thresholds and targets settings')
disp('--------------------------------------------------')

AHIthr = 15; % AHI threshold
disp('--------------------------------------------------')
fprintf('Threshold value for OSAHS screening: %d \n',AHIthr);
disp('--------------------------------------------------')
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

% perform ROC analysis
disp('--------------------------------------------------')
disp('ROC analysis performing...')
disp('--------------------------------------------------')
[X.X1,Y.Y1,T.T1,AUC.AUC1] = perfcurve(labels,AHIest1,'enfermo');
[X.X2,Y.Y2,T.T2,AUC.AUC2] = perfcurve(labels,AHIest2,'enfermo');
[X.X3,Y.Y3,T.T3,AUC.AUC3] = perfcurve(labels,AHIest3,'enfermo');
[X.X4,Y.Y4,T.T4,AUC.AUC4] = perfcurve(labels,AHIest4,'enfermo');
[X.X5,Y.Y5,T.T5,AUC.AUC5] = perfcurve(labels,AHIest5,'enfermo');
[X.X6,Y.Y6,T.T6,AUC.AUC6] = perfcurve(labels,AHIest6,'enfermo');
[X.X7,Y.Y7,T.T7,AUC.AUC7] = perfcurve(labels,AHIest7,'enfermo');
[X.X8,Y.Y8,T.T8,AUC.AUC8] = perfcurve(labels,AHIest8,'enfermo');
[X.X9,Y.Y9,T.T9,AUC.AUC9] = perfcurve(labels,AHIest9,'enfermo');
[X.X10,Y.Y10,T.T10,AUC.AUC10] = perfcurve(labels,AHIest10,'enfermo');
[X.X11,Y.Y11,T.T11,AUC.AUC11] = perfcurve(labels,AHIest11,'enfermo');
[X.X12,Y.Y12,T.T12,AUC.AUC12] = perfcurve(labels,AHIest12,'enfermo');
[X.X13,Y.Y13,T.T13,AUC.AUC13] = perfcurve(labels,AHIest13,'enfermo');
[X.X14,Y.Y14,T.T14,AUC.AUC14] = perfcurve(labels,AHIest14,'enfermo');
[X.X15,Y.Y15,T.T15,AUC.AUC15] = perfcurve(labels,AHIest15,'enfermo');
[X.X16,Y.Y16,T.T16,AUC.AUC16] = perfcurve(labels,AHIest16,'enfermo');
[X.X17,Y.Y17,T.T17,AUC.AUC17] = perfcurve(labels,AHIest17,'enfermo');
[X.X18,Y.Y18,T.T18,AUC.AUC18] = perfcurve(labels,AHIest18,'enfermo');
[X.X19,Y.Y19,T.T19,AUC.AUC19] = perfcurve(labels,AHIest19,'enfermo');
[X.X20,Y.Y20,T.T20,AUC.AUC20] = perfcurve(labels,AHIest20,'enfermo');
[X.X21,Y.Y21,T.T21,AUC.AUC21] = perfcurve(labels,AHIest21,'enfermo');
[X.X22,Y.Y22,T.T22,AUC.AUC22] = perfcurve(labels,AHIest22,'enfermo');
[X.X23,Y.Y23,T.T23,AUC.AUC23] = perfcurve(labels,AHIest23,'enfermo');
[X.X24,Y.Y24,T.T24,AUC.AUC24] = perfcurve(labels,AHIest24,'enfermo');
[X.X25,Y.Y25,T.T25,AUC.AUC25] = perfcurve(labels,AHIest25,'enfermo');
[X.X26,Y.Y26,T.T26,AUC.AUC26] = perfcurve(labels,AHIest26,'enfermo');
[X.X27,Y.Y27,T.T27,AUC.AUC27] = perfcurve(labels,AHIest27,'enfermo');
[X.X28,Y.Y28,T.T28,AUC.AUC28] = perfcurve(labels,AHIest28,'enfermo');
[X.X29,Y.Y29,T.T29,AUC.AUC29] = perfcurve(labels,AHIest29,'enfermo');
[X.X30,Y.Y30,T.T30,AUC.AUC30] = perfcurve(labels,AHIest30,'enfermo');
[X.X31,Y.Y31,T.T31,AUC.AUC31] = perfcurve(labels,AHIest31,'enfermo');
[X.X32,Y.Y32,T.T32,AUC.AUC32] = perfcurve(labels,AHIest32,'enfermo');
[X.X33,Y.Y33,T.T33,AUC.AUC33] = perfcurve(labels,AHIest33,'enfermo');
[X.X34,Y.Y34,T.T34,AUC.AUC34] = perfcurve(labels,AHIest34,'enfermo');
[X.X35,Y.Y35,T.T35,AUC.AUC35] = perfcurve(labels,AHIest35,'enfermo');
[X.X36,Y.Y36,T.T36,AUC.AUC36] = perfcurve(labels,AHIest36,'enfermo');
[X.X37,Y.Y37,T.T37,AUC.AUC37] = perfcurve(labels,AHIest37,'enfermo');
[X.X38,Y.Y38,T.T38,AUC.AUC38] = perfcurve(labels,AHIest38,'enfermo');
[X.X39,Y.Y39,T.T39,AUC.AUC39] = perfcurve(labels,AHIest39,'enfermo');
[X.X40,Y.Y40,T.T40,AUC.AUC40] = perfcurve(labels,AHIest40,'enfermo');
[X.X41,Y.Y41,T.T41,AUC.AUC41] = perfcurve(labels,AHIest41,'enfermo');

thresholds = [-0.2:0.01:0.2];
all_AUC = [AUC.AUC1 AUC.AUC2 AUC.AUC3 AUC.AUC4 AUC.AUC5 AUC.AUC6...
	AUC.AUC7 AUC.AUC8 AUC.AUC9 AUC.AUC10 AUC.AUC11 AUC.AUC12...
	AUC.AUC13 AUC.AUC14 AUC.AUC15 AUC.AUC16 AUC.AUC17 AUC.AUC18...
	AUC.AUC19 AUC.AUC20 AUC.AUC21 AUC.AUC22 AUC.AUC23 AUC.AUC24...
	AUC.AUC25 AUC.AUC26 AUC.AUC27 AUC.AUC28 AUC.AUC29 AUC.AUC30...
	AUC.AUC31 AUC.AUC32 AUC.AUC33 AUC.AUC34 AUC.AUC35 AUC.AUC36...
	AUC.AUC37 AUC.AUC38 AUC.AUC39 AUC.AUC40 AUC.AUC41];

[auc_max,thr_opt] = max(all_AUC);
disp('--------------------------------------------------')
fprintf(['El valor máximo de AUC para el método ' metodo ' es: %.4f \n'],auc_max)
fprintf(['El umbral óptimo para el método ' metodo ' es: %.2f \n'],thresholds(thr_opt))
disp('--------------------------------------------------')

% plots
disp('--------------------------------------------------')
disp('plots')
disp('--------------------------------------------------')

plot(thresholds,all_AUC,'k',thresholds(thr_opt),auc_max,'k*')
axis([-0.3 0.3 0 1])
grid on
xlabel('thresholds')
ylabel('AUC value')
%hold on


