clc
clear all
close all
% Downloading data
data_train = xlsread('Data1.xlsx','train');% importing train data 
data_test = xlsread('Data1.xlsx','test'); % importing test set

% Set the random number seed to make the results repeatable in this script
rng('default');

% Preparing the Data: Response and Predictors for the train set% %Response for training set
Ytrain = data_train(:,end);
disp('Gross national savings')
tabulate(Ytrain)
Ytrain = nominal(Ytrain);

% Predictor matrix for training set
Xtrain = data_train(:,1:end-1);

% Test set
% Response for test set
Ytest = data_test(:,end); 
disp('Gross national savings') 
tabulate(Ytest)
Ytest = nominal(Ytest);

% Predictor matrix for test set
Xtest = data_test(:,1:end-1);

% Model 1: Decision Trees
% growing classification tree
t = fitctree(Xtrain,Ytrain); 
view(t) % text description

% plotting tree
view(t,'Mode','graph')
% perform classification by using Statistics and Machine Learning ToolboxTM functions. Computing the resubstitution error and the cross-validation error for decision tree. 
% Using cvpartition to generate 10 disjoint stratified subsets.
cp = cvpartition(Ytrain,'k',10)
dtResubErr = resubLoss(t)
cvt = crossval(t,'CVPartition',cp); 
dtCVErr = kfoldLoss(cvt)

% Compute the resubstitution error for various subsets of the original tree %compute the cross-validation error for these sub-trees
% Plotting the comparisone plot of errors
resubcost = resubLoss(t,'Subtrees','all'); 
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all'); 
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes'); 
ylabel('Cost (misclassification error)') 
legend('Cross-validation','Resubstitution')

% As can be seen from the errors, tree overfits training set (the slope %between resub and cross-val errors increasing)
% Trying to find a simpler tree that performs better
% Than a more complex tree on new data
% We should choose the tree with the smallest cross-validation error
% take the simplest tree that is within one standard error of the minimum %That's the default rule used by the cvloss method of ClassificationTree
[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 100], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo') 
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice') 
hold off

% We can look at the pruned tree and compute the estimated misclassification error for it
pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')
cost(bestlevel+1) %ans = 0.2236
% This pruned tree (differs from default version by simplicity) can be used %for prediction on test set as well as original tree. Results will be %compared for the learning purpose (was it reasonable to prune the tree)
% Test from http://se.mathworks.com/help/stats/improving-classification-trees- and-regression-trees.html

% Selecting Appropriate Tree Depth
% Generating an exponentially spaced set of values from 10 through 100 %that represent the minimum number of observations per leaf node 
leafs = logspace(1,2,10);

% Creating cross-validated classification trees
% Specifying to grow each tree using a minimum leaf size in leafs 
rng('default')
N = numel(leafs);
err = zeros(N,1);
for n = 1:N
  t = fitctree(Xtrain,Ytrain,'CrossVal','On',...
          'MinLeafSize',leafs(n)); 
  err(n) = kfoldLoss(t);
end
figure
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');
%  The best leaf size is 12 observations per leaf

% Comparing the near-optimal tree with 12 observations per leaf with the default tree, which uses 10 observations per parent node and 1 observation per leaf
DefaultTree = t;

OptimalTree = fitctree(Xtrain,Ytrain,'MinLeafSize',12); 
view(OptimalTree,'mode','graph')

% Calcilating errors for comparison
resubOpt = resubLoss(OptimalTree);
lossOpt = kfoldLoss(crossval(OptimalTree)); 
resubDefault = resubLoss(DefaultTree); 
lossDefault = kfoldLoss(crossval(DefaultTree)); 
resubOpt,resubDefault,lossOpt,lossDefault

%resubOpt=0.1404, resubDefault=0.0501, lossOpt=0.2379, lossDefault=0.2104
% Near-optimal tree gives more higher resubstitutional error %and higher cross-validational error as default tree
% Next we trying to prune the Default tree
% Finding the optimal pruning level by minimizing cross-validated loss:
[~,~,~,bestlevel] = cvLoss(DefaultTree,... 'SubTrees','All','TreeSize','min')
%bestlevel = 9. Pruning the tree to level 9
view(DefaultTree,'Mode','Graph','Prune',9)
%Setting 'TreeSize' to 'SE' (default) to find the maximal pruning level for which the tree error does not exceed the error from the best level 
%plus one standard deviation
[~,~,~,bestlevel] = cvLoss(DefaultTree,'SubTrees','All') 
% best level = 13. Prunning the tree to the level 13. 
tree_def = prune(DefaultTree,'Level',13);
view(tree_def,'Mode','Graph')

%Same procedures for the test purpose we do for Optimal Tree
[~,~,~,bestlevel] = cvLoss(OptimalTree,'SubTrees','All','TreeSize','min')

[~,~,~,bestlevel] = cvLoss(DefaultTree,'SubTrees','All') %best level = 13. Prunning the tree to the level 13. tree_def = prune(DefaultTree,'Level',13); view(tree_def,'Mode','Graph')

% Same procedures for the test purpose we do for Optimal Tree
[~,~,~,bestlevel] = cvLoss(OptimalTree,... 'SubTrees','All','TreeSize','min')

%bestlevel = 2. Pruning the tree to level 2
view(OptimalTree,'Mode','Graph','Prune',2)

%Setting 'TreeSize' to 'SE' (default) to find the maximal pruning level %for which the tree error does not exceed the error from the best level 
%plus one standard deviation
[~,~,~,bestlevel] = cvLoss(OptimalTree,'SubTrees','All') 

% best level = 13. Prunning the tree to the level 13. 
tree_opt = prune(OptimalTree,'Level',13); 
view(tree_opt,'Mode','Graph')

% This two pruned trees can be used to predict on test set. However, we will use the original (default one) as well %and compare the perfomance

% Predicting Y on test set by using the first tree (model t)
Y_t = predict(t,Xtest);
% Compute the confusion matrix
C_t = confusionmat(Ytest,Y_t)
% Examine the confusion matrix for each class as a percentage of the true class
C_tperc = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100
%predicting Y on test set by using the first tree (model pt)
Y_pt = predict(pt,Xtest);
% Compute the confusion matrix
C_pt = confusionmat(Ytest,Y_pt)
% Examine the confusion matrix for each class as a percentage of the true class
C_ptperc = bsxfun(@rdivide,C_pt,sum(C_pt,2)) * 100
%predicting Y on test set by using the first tree (model tree_def)
Y_tree_def = predict(tree_def,Xtest);
% Compute the confusion matrix
C_tree_def = confusionmat(Ytest,Y_tree_def)
% Examine the confusion matrix for each class as a percentage of the true class
C_tree_defperc = bsxfun(@rdivide,C_tree_def,sum(C_pt,2)) * 100
%predicting Y on test set by using the first tree (model tree_def)
Y_tree_opt = predict(tree_opt,Xtest);

% Compute the confusion matrix
C_tree_opt = confusionmat(Ytest,Y_tree_opt)
% Examine the confusion matrix for each class as a percentage of the true class
C_tree_optperc = bsxfun(@rdivide,C_tree_opt,sum(C_pt,2)) * 100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----------------%%%%%%%%%%% 
% Model 2:Support Vector Machines
opts = statset;
% Training the classifier
svmStruct = svmtrain(Xtrain,Ytrain,'kernel_function','rbf','kktviolationlevel',0.1,'optio ns',opts);
% Making  a prediction for the test set
Y_svm = svmclassify(svmStruct,Xtest);
C_svm = confusionmat(Ytest,Y_svm);
C_svmperc = bsxfun(@rdivide,C_svm,sum(C_svm,2)) * 100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----------------%%%%%%%%%%%
% Model 3: Logistic Regression
% fitting the model with function glm
glm = fitglm(Xtrain,Ytrain,'linear','Distribution','binomial','link','logit'); 
%not all variables have p-value < 0.5. Removing x3 for improving the model 
glm1 = removeTerms(glm,'x3');

plotSlice(glm);

% stepwise model
sw_glm = stepwiseglm(Xtrain,Ytrain,'linear','Distribution','binomial','link','logit'); 
sw_glm1 = stepwiseglm(Xtrain,Ytrain,'constant','Distribution','binomial','upper','linea r');

% Prediction for test set with glm
Y_glm = predict(glm,Xtest);
Y_glm = round(Y_glm);
% Compute the confusion matrix
C_glm = confusionmat(double(Ytest),Y_glm);
% Pediction for test set with sw_glm
Y_swglm = predict(sw_glm,Xtest);
Y_swglm = round(Y_swglm);
% Compute the confusion matrix
C_swglm = confusionmat(double(Ytest),Y_swglm);
% Prediction for test set with glm
Y_glm1 = predict(glm1,Xtest); Y_glm1 = round(Y_glm1);

% Compute the confusion matrix
C_glm1 = confusionmat(double(Ytest),Y_glm1);
% Examine the confusion matrix for each class as a percentage of the true class
C_glmperc = bsxfun(@rdivide,C_glm,sum(C_glm,2)) * 100
C_glmperc1 = bsxfun(@rdivide,C_glm1,sum(C_glm1,2)) * 100
C_swglmperc = bsxfun(@rdivide,C_swglm,sum(C_swglm,2)) * 100
% Stage 2
% Compare Results
Cmat = [ C_svmperc C_ptperc C_swglmperc ];
labels = {'Support VM ', 'Decision Trees ', 'Logistic Regression '}; comparisonPlot( Cmat, labels )
