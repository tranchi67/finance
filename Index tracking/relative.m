function rs=relative(x)
    quanth=[(order(x(:,1)))' (order(x(:,2)))'];
    [m n]=size(quanth);
    for i=1:10
        for j=1:10
            count=0;
            for k=1:m
                if quanth(k,:)==[i,j]
                    count=count+1;
                else count=count+0;
                end
            end
            a(i,j)=count*10/m;
        end
    end
    for i=1:10
        D(i)=sum(sum(a(1:i,1:i)));
    end
    rs=0;
    for i=1:10
        rs=rs+1/10*(i/10-D(i)/10);
    end
end