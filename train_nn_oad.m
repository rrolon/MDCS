% train neural network using overcomplete dictionary

% neural network training using mini-batch

% metodo = 'FULL-OAD';
% metodo = 'MDAS-OAD';
metodo = 'MDCS-OAD'

disp('--------------------------------------------------')
disp(['Method: ' metodo])
disp('--------------------------------------------------')

disp('--------------------------------------------------')
disp('neural network training using mini-batch')
disp('--------------------------------------------------')

% dictionary assembly

L = 16; % number of non-zero elements 
number_of_atoms = 64;
main_indices = [indices1(1:number_of_atoms/2) indices2(1:number_of_atoms/2)];
% A = Phi_OAD; % use in case of FULL-OAD and MDAS-OAD methods
A = Phi_OAD(:,main_indices); % sub-dictionary in case of MDCS-OAD method

inputs = full(omp(A,X_1_667,[],L));; % inputs of the neural network
% inputs = inputs(main_indices,:); % use in case of MDAS-OAD method
targets = 2*T_1_667-1; % 1 -> apnea event and -1 -> normal
[segment_size,segments] = size(inputs);
batch_size = 1000; % size of mini batch
num_epochs = floor(segments/batch_size);
numHiddenNeurons = 32;  % Adjust as desired

disp('--------------------------------------------------')
fprintf('Number of non-zero elements: %d \n',L);
fprintf('Number of atoms: %d \n',number_of_atoms);
fprintf('Number of neurons in hidden layer: %d \n',numHiddenNeurons);
disp('--------------------------------------------------')

disp('--------------------------------------------------')
fprintf('Mini-batch size: %d \n',batch_size);
fprintf('Number of epochs (iterations): %d \n',num_epochs);
disp('--------------------------------------------------')

% divide inputs into inputs class 1 and inputs class 2
I = find(T_1_667); % indices corresponding to class 1
inputs_c1 = inputs(:,I); % inputs class 1
[nc1,mc1] = size(inputs_c1); % mc1 represents the number of segments corresponding to class 1
inputs(:,I) = [];
inputs_c2 = inputs; % inputs class 2
[nc2,mc2] = size(inputs_c2); % mc2 represents the number of segments corresponding to class 2
disp('--------------------------------------------------')
fprintf('Number of segments corresponding to class 1: %d \n',mc1);
fprintf('Number of segments corresponding to class 2: %d \n',mc2);
disp('--------------------------------------------------')

% create neural network
net = newfit(inputs,targets,numHiddenNeurons);

% configure neural network
net.divideParam.trainRatio = 70/100;  % Adjust as desired
net.divideParam.valRatio = 15/100;  % Adjust as desired
net.divideParam.testRatio = 15/100;  % Adjust as desired
net.trainParam.epochs = 4;% num_epochs; % Maximum number of epochs to train
net.trainParam.goal = 0.001; % Performance goal
net.trainParam.lr = 0.01; % Learning rate
% net.trainParam.lr_inc = 1.05; % Ratio to increase learning rate
net.trainParam.max_fail = 5; % Maximum validation failures
net.trainParam.min_grad = 1e-5; % Minimum performance gradient
net.trainParam.show = 1; % Epochs between displays (NaN for no displays)
net.trainParam.showWindow = false; % Show training GUI

% training neural network
disp('--------------------------------------------------')
disp('training neural network')
disp('--------------------------------------------------')
for iter = 1:num_epochs
	fprintf('Iteration number %d \n',iter);
	ind_of_segments_c1 = rand_sin_repeticion(1,mc1,batch_size/2);
	ind_of_segments_c2 = rand_sin_repeticion(1,mc2,batch_size/2);
	inpt_c1 = inputs_c1(:,ind_of_segments_c1);
	inpt_c2 = inputs_c2(:,ind_of_segments_c2);
	inpt = [inpt_c1 inpt_c2];
	targt = [ones(1,batch_size/2) -ones(1,batch_size/2)];
	[net,tr] = train(net,inpt,targt);
end

disp('--------------------------------------------------')
disp('testing neural network')
disp('--------------------------------------------------')
ind_of_tst_segments_c1 = rand_sin_repeticion(1,mc1,batch_size/2);
ind_of_tst_segments_c2 = rand_sin_repeticion(1,mc2,batch_size/2);
inpt_tst_c1 = inputs_c1(:,ind_of_tst_segments_c1);
inpt_tst_c2 = inputs_c2(:,ind_of_tst_segments_c2);
inpt_tst = [inpt_tst_c1 inpt_tst_c2];
out_tst = sim(net,inpt_tst);

% batch learning of the neural network

%inputs1 = [alpha(:,1:10:500) alpha(:,2:10:500)];
%targets1 = [(2*T_1_667(1:10:500)-1) (2*T_1_667(2:10:500)-1)];
%outputs1 = sim(net,inputs1);


