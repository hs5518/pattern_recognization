clear;
clc;
close all;

%load the data
load('F0_PVT.mat');

%standardize data
two_data = [acrylic steel_vase];
two_data = standardize(two_data);

acrylic_data = two_data(:,1:10);
steel_vase_data = two_data(:,11:20);

figure;
scatter3(acrylic_data(1,:),acrylic_data(2,:),acrylic_data(3,:),'green','filled');
hold on;
scatter3(steel_vase_data(1,:),steel_vase_data(2,:),steel_vase_data(3,:),'magenta','filled');
xlabel('pressure');
ylabel('vibration');
zlabel('temperature');

[Sw,Sb] = lda_n(two_data);
[V,D,U] = eig(Sw\Sb);
Z = zeros(1,3);
V_1 = V(:,1)';
V_2 = V(:,2)';
V_3 = V(:,3)';
pts1 = [Z; V_1*2];
pts2 = [Z; V_2*2];
pts3 = [Z; V_3*2];
plot3(pts1(:,1), pts1(:,2), pts1(:,3))
plot3(pts2(:,1), pts2(:,2), pts2(:,3))
plot3(pts3(:,1), pts3(:,2), pts3(:,3))

%% ------------------



% lambda=diag(D);
% [lambda, SortOrder]=sort(lambda,'descend');
% V=V(:,SortOrder);

acrylic_data = acrylic_data' * V;
steel_vase_data = steel_vase_data' * V;

%plot the projection
figure;
subplot(3,1,1);
%pressure
scatter(acrylic_data(:,1),0,"black",'*');
hold on;
scatter(mean(acrylic_data(:,1)),0,'black');
scatter(steel_vase_data(:,1),0,"red",'*');
scatter(mean(steel_vase_data(:,1)),0,"red");
title('LDA component 1');
%vibration
subplot(3,1,2);
scatter(acrylic_data(:,2),0,"black",'*');
hold on;
scatter(mean(acrylic_data(:,2)),0,'black');
scatter(steel_vase_data(:,2),0,"red",'*');
scatter(mean(steel_vase_data(:,2)),0,"red");
title('LDA component 2');
%temperature
subplot(3,1,3);
scatter(acrylic_data(:,3),0,"black",'*');
hold on;
scatter(mean(acrylic_data(:,3)),0,'black');
scatter(steel_vase_data(:,3),0,"red",'*');
scatter(mean(steel_vase_data(:,3)),0,"red");
title('LDA component 3');