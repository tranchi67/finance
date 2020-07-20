function as=absolute(x,y)
    [r c]=size(x);
    for i=1:r
        for j=1:c
            as(i,j)=y(i,j)/x(i,j);
        end
    end
end
