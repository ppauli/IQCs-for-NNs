function [W,b]=generate_random_NN(n_vec)

l=length(n_vec)-1;

for ii=1:l
    W{ii}=randn(n_vec(ii+1),n_vec(ii));
    b{ii}=randn(n_vec(ii+1),1);
end