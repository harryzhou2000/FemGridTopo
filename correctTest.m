
i = [];
j = [];
v = [];
N = 10000*1000;
for it = 1:N
    if(it>1)
        i(end+1) = it;
        j(end+1) = it-1;
        v(end+1) = -1;
    end
    if(it<N)
        i(end+1)=it;
        j(end+1) = it+1;
        v(end+1) = -1;
    end
    i(end+1) = it;
    j(end+1) = it;
    v(end+1) = 2;
end
i(end+1) = 1;
j(end+1) = 1;
v(end+1) = 123;
b = ones(N,1);
A = sparse(i,j,v,N,N);
%%
tic;
for it = 1:10
    x = (A)\b;
end
toc
