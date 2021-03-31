function y=Leaky_ReLU(x)

for ii=1:size(x,1)
    for jj=1:size(x,2)
        if x(ii,jj)<0
            y(ii,jj)=0.1*x(ii,jj);
        else
            y(ii,jj)=x(ii,jj);
        end
    end
end