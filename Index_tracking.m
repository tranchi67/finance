clc;
clear all;
close all;

data=xlsread("data_the2.xlsx");% load index price and stocks prices
period_1=data(1:260,:); 
[m n] = size(period_1); 
t = 65;
for i = 1:m-1 % compute returns 
    for j = 1:n
    returns(i,j) = (period_1(i+1,j)-period_1(i,j))/period_1(i,j);
    end 
end

stock = returns(:,2:end); 
[row col] = size(stock);
index = returns(:,1);
e=[1:10];

for i=1:10 
    C(i)=10^(-1*e(i));
end
K = cell([4 1]);% create k folds for cross-validation 
for i=1:4
    K{1} = [1 t-1]; 
    K{2} = [t 2*t]; 
    K{3} = [2*t+1 3*t]; 
    K{4} = [3*t+1 row];
end

A=[5:5:25];% number of asset MSE_train=[];
MSE_test=[];

% Model Selection
for i = 1:length(C) % run model with different values of lambda 
    c = C(i);
    for j=1:length(A) 
        a = A(j);
        x0=1/a*ones(a,1);
        for s = 1:1000 % 1000 random portfolio
            p = randperm(n-1,a);
            stock_p = stock(:,p);
            for k = 1:4 % run on different training sets P = K{k};
                train =[1:P(1)-1 P(2)+1:row];
                x_train = stock_p(train,:);
                x_test = stock_p(P(1):P(2),:);
                y_train = index(train); y_test = index(P(1):P(2));
                % Find weights of assets by the Ridge regression:
                rng default
                gs = GlobalSearch('Display','final'); options =
                optimoptions('fmincon','Display','off','Algorithm','sqp'); problem.options = options;
                problem.solver = 'fmincon';
                problem.objective = @(w)sum((y_train- x_train*w).^2)+c*sum(w.^2);
                problem.lb = zeros(a,1); problem.ub = ones(a,1); problem.Aeq = ones(1,a); problem.beq = [1]; problem.x0=x0; weight=run(gs,problem);
                
                % Compute tracking quality
                RMSE_p_train(k) = sqrt(mean((y_train- x_train*weight).^2))*100;
                RMSE_p_test(k) = sqrt(mean((y_test- x_test*weight).^2))*100;
            end
            
            RMSE_c_train(s,j) = mean(RMSE_p_train);
            RMSE_c_test(s,j) = mean(RMSE_p_test);% tracking quality of each portfolio
        end 
    end

    RMSE_train = [RMSE_train mean(RMSE_c_train)];
    RMSE_test = [RMSE_test mean(RMSE_c_test)];
end
R = [mean(RMSE_train') mean(RMSE_test')]

%% TEV model

t = 195;
K = (5:5:25); % portfolio with k assets 
MSE_train = zeros(1000,length(K)); 
MSE_test = zeros(1000,length(K));

for a = 1:length(K) % run model with different value of k 
    k = K(a);
    for p = 1:1000 % select 1000 portfolio randomly 
        P = randperm(n-1,k);
        x_train = stock(1:t,P); 
        x_test = stock(t+1:end,P); 
        y_train = index(1:t); 
        y_test = index(t+1:end);
        % Find the weights of assets (tracking error optimization model) by     minimizing TEV
        w= optimvar('w',[k,1],'Type','integer','LowerBound',0,'UpperBound',1); 
        objec = mean((y_train-x_train*w-mean(y_train)+mean(x_train*w)).^2); 
        prob = optimproblem('Objective',objec);
        prob.Constraints.cons = sum(w) == 1;
        problem = prob2struct(prob); weight = quadprog(problem);

        % Compute tracking quality
        MSE_train(p,a) = mean((y_train-x_train*weight).^2);
        MSE_test(p,a) = mean((y_test-x_test*weight).^2);
    end
end

RMSE_train = sqrt(MSE_train)*100; 
RMSE_test = sqrt(MSE_test)*100;

% Stability level
ab_stability = absolute(RMSE_train,RMSE_test);% absolute stabilty
ab_sta = [median(ab_stability) 
    max(ab_stability) 
    min(ab_stability)]

c1 = cell([1 5]); 
for i=1:5
    c{i} = [RMSE_train(:,i) RMSE_test(:,i)];
end

for i=1:5 % compute relative stability by function relative 
    rel_stability(i) = relative(c{i});
end

% Display result
A=[median(RMSE_train) 
    median(RMSE_test) 
    mean(RMSE_train) 
    mean(RMSE_test)
    std(RMSE_train) 
    std(RMSE_test) 
    max(RMSE_train) 
    max(RMSE_test) 
    min(RMSE_train) 
    min(RMSE_test)]


%% The Ridge regression model
for a = 1:length(K) % run model with different value of k 
    k = K(a);
    x0=1/k*ones(k,1);
    for p = 1:1000 % select 1000 portfolio randomly
        P = randperm(n-1,k);
        x_train = stock(1:t,P);
        x_test = stock(t+1:end,P);
        y_train = index(1:t);
        y_test = index(t+1:end);
        
        % Find the weights of assets
        rng default
        gs = GlobalSearch('Display','final');
        options = optimoptions('fmincon','Display','off','Algorithm','sqp'); problem.options = options;
        problem.solver = 'fmincon';
        problem.objective = @(w)sum((y_train-x_train*w).^2)+10^(-8)*sum(w.^2); problem.lb = zeros(k,1);
        problem.ub = ones(k,1);
        problem.Aeq = ones(1,k);
        problem.beq = [1];
        problem.x0=x0;
        weight=run(gs,problem);
        
        % Compute tracking quality
        MSE_train(p,a) = mean((y_train-x_train*weight).^2);
        MSE_test(p,a) = mean((y_test-x_test*weight).^2);
        RMSE_train = sqrt(MSE_train)*100; RMSE_test = sqrt(MSE_test)*100;
    end
end

% Stability level
ab_stability = absolute(RMSE_train,RMSE_test);% absolute stabilty
ab_sta =[median(ab_stability) 
    max(ab_stability) 
    min(ab_stability)]
c1 = cell([1 5]); 

for i=1:5
    c{i} = [RMSE_train(:,i) 
    RMSE_test(:,i)];
end
for i=1:5 % compute relative stability by function relative 
    rel_stability(i) = relative(c{i});
end

% Display result
A=[median(RMSE_train)
   median(RMSE_test) 
   mean(RMSE_train) 
   mean(RMSE_test)
   std(RMSE_train) 
   std(RMSE_test) 
   max(RMSE_train) 
   max(RMSE_test) 
   min(RMSE_train) 
   min(RMSE_test)]