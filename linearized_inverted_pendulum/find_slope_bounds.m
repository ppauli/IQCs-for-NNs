function [alpha_phi,beta_phi]=find_slope_bounds(W,b,xs,dv,act)

l=length(W)-1;

vs{1}=W{1}*xs+b{1};
v_max{1}=vs{1}+dv{1};
v_min{1}=vs{1}-dv{1};

for ii=2:l % Interval bound propagation
    vs{ii}=W{ii}*tanh(vs{ii-1})+b{ii};
    mu=(tanh(v_max{ii-1})+tanh(v_min{ii-1}))/2;
    r=(tanh(v_max{ii-1})-tanh(v_min{ii-1}))/2;
    mu2=W{ii}*mu+b{ii};
    r2=abs(W{ii})*r;
    v_min{ii}=mu2-r2;
    v_max{ii}=mu2+r2;
end
%%

if strcmp(act,'ReLU')
for jj=1:l
    for kk=1:size(W{jj},1)
        if ReLU(v_max{jj}(kk))-ReLU(vs{jj}(kk))==0
            beta_phi{jj}(kk)=0;
        else
            beta_phi{jj}(kk)=1;
        end
        if ReLU(v_min{jj}(kk))-ReLU(vs{jj}(kk))==v_min{jj}(kk)-vs{jj}(kk)
            alpha_phi{jj}(kk)=1;
        else
            alpha_phi{jj}(kk)=0;
        end        
    end
end
elseif strcmp(act,'L_ReLU')
for jj=1:l
    for kk=1:size(W{jj},1)
        if ReLU(v_max{jj}(kk))-ReLU(vs{jj}(kk))==v_max{jj}(kk)-vs{jj}(kk)
            beta_phi{jj}(kk)=1;
        else
            beta_phi{jj}(kk)=0.1;
        end
        if ReLU(v_min{jj}(kk))-ReLU(vs{jj}(kk))==v_min{jj}(kk)-vs{jj}(kk)
            alpha_phi{jj}(kk)=1;
        else
            alpha_phi{jj}(kk)=0.1;
        end        
    end
end
else
for jj=1:l
    for kk=1:size(W{jj},1)
        if sign(v_min{jj}(kk)*v_max{jj}(kk))==-1
            beta_phi{jj}(kk)=1;
            if abs(v_min{jj}(kk))>v_max{jj}(kk)
                alpha_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2; % derivative of tanh
            else
                alpha_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
            end
        else
            if sign(v_min{jj}(kk))==-1
                alpha_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2;
                beta_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
            else
                alpha_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
                beta_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2;
            end            
        end
    end
end
end
end