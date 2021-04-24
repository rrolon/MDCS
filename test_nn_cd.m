% test neural network using complete dictionary

%% Toolboxes
t = genpath('/home/rrolon/Sleep Apnea/toolboxes'); 
addpath(t); % se carga los toolboxes

%% definitions
studies_location = ['/home/rrolon/Sleep Apnea/NOCICA_MDAS_NEW/defs/all_data_without_errors.txt']; % link data

studies_identification = textread(studies_location,'%s');
n_studies = length(studies_identification); % number of studies
n_studies_70 = floor(0.7*n_studies);
n_studies_30 = n_studies-n_studies_70;
%indices_train = rand_sin_repeticion(1,n_studies,n_studies_70);
%indices_test = get_test_indices(indices_train,n_studies);

L = 16; % number of non-zero elements
NOVER = 1;    % frame overlap
NWIN = 128;  % cantidad de muestras de los frames para formar matriz de
total_frames = [];

for i = 1:n_studies_30
    disp(['iteration ' num2str(i) ' loading PSG ' num2str(indices_test(i)) ' ' char(studies_identification(i))]);
    psg = char(studies_identification(indices_test(i)));
    load(['/home/rrolon/Sleep Apnea/SHHS_database/' psg],'Flujo','SaO2','apn','SleepStage');
    apn = apn(1:10:end);
    Flujo.signal = Flujo.signal(1:10:end);
    [spo2, apn_spo2] = filtrado1(SaO2.signal, apn);
    n = length(spo2); % cantidad de muestras de la señal spo2
    frames = floor((n-NWIN)/NOVER);
    total_frames = [total_frames frames];
    X = []; % matriz de señales
    TARG = []; % matriz de targets para entrenar
    j = 1;
    while j <= frames;
        aux = apn_spo2(1+NOVER*j:NWIN+j*NOVER);
	if max(aux)<300
		TARG = [TARG max(aux)>0];
   	        X = [X spo2(1+NOVER*j:NWIN+j*NOVER)];
	end
        j = j+1;
    end % end while   
    time_sleep(i) = sum(SleepStage > 0)/10/60/60;
    AHI(i)=sum((diff(apn==100 |apn==200) ~= 0))/2/time_sleep(i);
    alfa = full(omp(A,X,[],L)); % sparse coefficient matrix     
    alfa = alfa(main_indices,:); % use in case of MDAS-CD method
    y = net(alfa);    
    AHIest_n1(i) = get_AHIest(y,time_sleep(i),1);    
    AHIest_n2(i) = get_AHIest(y,time_sleep(i),2);    
    AHIest_n3(i) = get_AHIest(y,time_sleep(i),3);    
    AHIest_n4(i) = get_AHIest(y,time_sleep(i),4);    
    AHIest_n5(i) = get_AHIest(y,time_sleep(i),5);    
    AHIest_n6(i) = get_AHIest(y,time_sleep(i),6);    
    AHIest_n7(i) = get_AHIest(y,time_sleep(i),7);    
    AHIest_n8(i) = get_AHIest(y,time_sleep(i),8);    
    AHIest_n9(i) = get_AHIest(y,time_sleep(i),9);    
    AHIest_n10(i) = get_AHIest(y,time_sleep(i),10);    
    AHIest_n11(i) = get_AHIest(y,time_sleep(i),11);    
    AHIest_n12(i) = get_AHIest(y,time_sleep(i),12);    
    AHIest_n13(i) = get_AHIest(y,time_sleep(i),13);    
    AHIest_n14(i) = get_AHIest(y,time_sleep(i),14);    
    AHIest_n15(i) = get_AHIest(y,time_sleep(i),15);    
    AHIest_n16(i) = get_AHIest(y,time_sleep(i),16);    
    AHIest_n17(i) = get_AHIest(y,time_sleep(i),17);    
    AHIest_n18(i) = get_AHIest(y,time_sleep(i),18);    
    AHIest_n19(i) = get_AHIest(y,time_sleep(i),19);    
    AHIest_n20(i) = get_AHIest(y,time_sleep(i),20);    
    AHIest(i) = sum(abs(diff(y>0))/2)/time_sleep(i);
    AHIest1(i) = sum(y>-0.2)/time_sleep(i);    
    AHIest2(i) = sum(y>-0.19)/time_sleep(i);
    AHIest3(i) = sum(y>-0.18)/time_sleep(i);
    AHIest4(i) = sum(y>-0.17)/time_sleep(i);    
    AHIest5(i) = sum(y>-0.16)/time_sleep(i);
    AHIest6(i) = sum(y>-0.15)/time_sleep(i);
    AHIest7(i) = sum(y>-0.14)/time_sleep(i);    
    AHIest8(i) = sum(y>-0.13)/time_sleep(i);
    AHIest9(i) = sum(y>-0.12)/time_sleep(i);
    AHIest10(i) = sum(y>-0.11)/time_sleep(i);    
    AHIest11(i) = sum(y>-0.10)/time_sleep(i);
    AHIest12(i) = sum(y>-0.09)/time_sleep(i);
    AHIest13(i) = sum(y>-0.08)/time_sleep(i);    
    AHIest14(i) = sum(y>-0.07)/time_sleep(i);
    AHIest15(i) = sum(y>-0.06)/time_sleep(i);
    AHIest16(i) = sum(y>-0.05)/time_sleep(i);    
    AHIest17(i) = sum(y>-0.04)/time_sleep(i);
    AHIest18(i) = sum(y>-0.03)/time_sleep(i);
    AHIest19(i) = sum(y>-0.02)/time_sleep(i);    
    AHIest20(i) = sum(y>-0.01)/time_sleep(i);
    AHIest21(i) = sum(y>0)/time_sleep(i);
    AHIest22(i) = sum(y>-0.01)/time_sleep(i);    
    AHIest23(i) = sum(y>-0.02)/time_sleep(i);
    AHIest24(i) = sum(y>-0.03)/time_sleep(i);
    AHIest25(i) = sum(y>-0.04)/time_sleep(i);    
    AHIest26(i) = sum(y>-0.05)/time_sleep(i);
    AHIest27(i) = sum(y>-0.06)/time_sleep(i);
    AHIest28(i) = sum(y>-0.07)/time_sleep(i);    
    AHIest29(i) = sum(y>-0.08)/time_sleep(i);
    AHIest30(i) = sum(y>-0.09)/time_sleep(i);
    AHIest31(i) = sum(y>-0.10)/time_sleep(i);    
    AHIest32(i) = sum(y>-0.11)/time_sleep(i);
    AHIest33(i) = sum(y>-0.12)/time_sleep(i);
    AHIest34(i) = sum(y>-0.13)/time_sleep(i);    
    AHIest35(i) = sum(y>-0.14)/time_sleep(i);
    AHIest36(i) = sum(y>-0.15)/time_sleep(i);
    AHIest37(i) = sum(y>-0.16)/time_sleep(i);    
    AHIest38(i) = sum(y>-0.17)/time_sleep(i);
    AHIest39(i) = sum(y>-0.18)/time_sleep(i);
    AHIest40(i) = sum(y>-0.19)/time_sleep(i);    
    AHIest41(i) = sum(y>-0.20)/time_sleep(i);
clear X TARG alfa y
end % end for i
