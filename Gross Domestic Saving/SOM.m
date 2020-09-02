clc
clear all
close all

% loading data
[D_num,D_txt]=xlsread('data2.xlsx'); % reading the data from the excel file 
[d1,d2]=size(D_txt); % recording the size of the total datatable 
labs=D_txt(2:d1,1); % extracting labels of observations for the future SOM 
vnames=D_txt(1,2:d2); % extracting variable names 
sD=som_data_struct(D_num,'labels',labs,'comp_names',vnames);

% Normalization
sD_norm=som_normalize(sD,'logistic');

% Making SOM
sMap=som_make(sD_norm);
sMap = som_autolabel(sMap,sD_norm,'vote');

% Vizualize & Analyze
    % Data statistics
    csS = som_stats(sMap);
    som_table_print(som_stats_table(csS)); % showing the table of stats
    
    % U-matrix and component planes
    figure
    som_show(sMap);
    % preparation for hits & lables & clusering
    [Pd,V,me,l] = pcaproj(sD_norm,2); 
    Pm = pcaproj(sMap,V,me); % PC- projection
    Code = som_colorcode(Pm); % color coding
    hits = som_hits(sMap,sD_norm); % hits
    U = som_umat(sMap); % U-matrix
    Dm = U(1:2:size(U,1),1:2:size(U,2)); % distance matrix
    Dm = 1-Dm(:)/max(Dm(:)); Dm(find(hits==0)) = 0; % clustering info
    
    % Hits & labels
        figure
        % color code (distance) and hits
    subplot(1,2,1)
    som_cplane(sMap,Code,Dm);
    hold on 
    som_grid(sMap,'Label',cellstr(int2str(hits)),...
                'Line','none','Marker','none','Labelcolor','k'); 
    hold off
    title('Color code & hits')
    
    % lables
    subplot(1,2,2)
    som_cplane(sMap,'none')
    hold on 
    som_grid(sMap,'Label',sMap.labels,'Labelsize',8,...
                    'Line','none','Marker','none','Labelcolor','r');
    hold off
    title('Labels')
    
% Clustering
figure
% optimal number of clusters
subplot(1,3,1)
[c,p,err,ind] = kmeans_clusters(sMap, 7); % find at most n clusters. The optimal amount of clusters has the lowest k
                                          % (you might want to increase n to allow more clusters)
plot(1:length(ind),ind,'x-') 
[dummy,i] = min(ind);
cl = p{i};
title('Optimal # of clusters') 
% distance matrix s
ubplot(1,3,2) 
som_cplane(sMap,Code,Dm) 
title('Distance matrix')
% clusters
subplot(1,3,3)
som_cplane(sMap,cl)
title('Clusters')

% Statistics of clusters (optional does not work properly with missing values)
[V,I]=som_divide(sMap, sD, cl); % dividing SOM into defined earlier clusters
for j=1:i
    disp(strcat('cluster_',num2str(j))); % displaying cluster # 
    csS_cl=som_stats(V{j}); % getting stats struct from data cluster 
    for k=1:size(vnames,2) % this loop adds variable names
        csS_cl{k}.name=vnames{k};
    end
stats_cl=som_stats_table(csS_cl); % creating table of stats 
som_table_print(stats_cl); % printing table in command window 
disp('______________________________________________');
end
