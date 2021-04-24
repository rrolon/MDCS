function [aux3_ordered,indices] = freq_act(alfa,T)
%   year: 2016
%   Author: Roman Rolon
%   Research institute for Signals, Systems and Computational Intelligence sinc(i)
%   http://www.sinc.santafe-conicet.gov.ar/
%   rrolon@sinc.unl.edu ar
% Description:
%   The feature extraction is obtained by computing the atom activation 
%   frequency given the class here being (with and without AH). The 
%   candidates to be considered as input of the NN are then those atoms with
%   higher absolute between frequency activation for each of the clases. 
%   That is, if some atom is active many times for signals with AH events 
%   than for the signals without AH events, it is taken into account.
% Parameter:
%   alfa coefficient matrix
%   T targets
% Recommendation:
%   ...
% Caution:
%   ...
    L = 16;
    [n_atoms,n_segments] = size(alfa);
    aux1 = zeros(1,n_atoms);
    aux2 = zeros(1,n_atoms);
    for i = 1:n_segments
        for j = 1:n_atoms
            if T(i) == 0 % class 2
                if abs(alfa(j,i)) > 0 % threshold
                    aux1(1,j) = aux1(1,j)+1;
                end
            end
            if T(i) == 1 % class 1
                if abs(alfa(j,i)) > 0 % threshold
                    aux2(1,j) = aux2(1,j)+1;
                end
            end
        end % end for j
    end % end for i
    aux3 = abs(aux1-aux2);
    [aux3_ordered,indices] = sort(aux3,'descend');
end
