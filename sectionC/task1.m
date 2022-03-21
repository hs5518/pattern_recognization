clear;
clc;
close all;
%part a
%load the data
load('F0_PVT.mat');

%standardize data
two_data = [black_foam car_sponge];
two_data = standardize(two_data);

black_foam_pressure = two_data(1,1:10);
black_foam_vibration = two_data(2,1:10);
black_foam_temperature = two_data(3,1:10);
car_sponge_pressure = two_data(1,11:20);
car_sponge_vibration = two_data(2,11:20);
car_sponge_temperature = two_data(3,11:20);

%get mean for black foam and car sponge
black_foam_pressure_mean = mean(black_foam_pressure);
black_foam_vibration_mean = mean(black_foam_vibration);
black_foam_temperature_mean = mean(black_foam_temperature);

car_sponge_pressure_mean = mean(car_sponge_pressure);
car_sponge_vibration_mean = mean(car_sponge_vibration);
car_sponge_temperature_mean = mean(car_sponge_temperature);

%plot the points
%Pressure vs vibration
figure;
subplot(1,3,1)
scatter(black_foam_pressure,black_foam_vibration,'filled','black');
hold on;
scatter(car_sponge_pressure,car_sponge_vibration,'filled','yellow');
xlabel('pressure');
ylabel('vibration');
%calculate SW and Sb
Sw = ([black_foam_pressure;black_foam_vibration] - [black_foam_pressure_mean;black_foam_vibration_mean]) * ([black_foam_pressure;black_foam_vibration] - [black_foam_pressure_mean;black_foam_vibration_mean])' + ([car_sponge_pressure;car_sponge_vibration] - [car_sponge_pressure_mean;car_sponge_vibration_mean]) * ([car_sponge_pressure;car_sponge_vibration] - [car_sponge_pressure_mean;car_sponge_vibration_mean])';
Sb = ([black_foam_pressure_mean;black_foam_vibration_mean] - [car_sponge_pressure_mean;car_sponge_vibration_mean]) * ([black_foam_pressure_mean;black_foam_vibration_mean] - [car_sponge_pressure_mean;car_sponge_vibration_mean])';
[V,D,U] = eig(Sw\Sb);
V_1 = V(:,1);
plot([-V_1(1),V_1(1)]*2.5,[V_1(2),-V_1(2)]*2.5);
%pressure vs temperature

subplot(1,3,2)
scatter(black_foam_pressure,black_foam_temperature,'filled','black');
hold on;
scatter(car_sponge_pressure,car_sponge_temperature,'filled','yellow');
xlabel('pressure');
ylabel('temperature');
Sw = ([black_foam_pressure;black_foam_temperature] - [black_foam_pressure_mean;black_foam_temperature_mean]) * ([black_foam_pressure;black_foam_temperature] - [black_foam_pressure_mean;black_foam_temperature_mean])' + ([car_sponge_pressure;car_sponge_temperature] - [car_sponge_pressure_mean;car_sponge_temperature_mean]) * ([car_sponge_pressure;car_sponge_temperature] - [car_sponge_pressure_mean;car_sponge_temperature_mean])';
Sb = ([black_foam_pressure_mean;black_foam_temperature_mean] - [car_sponge_pressure_mean;car_sponge_temperature_mean]) * ([black_foam_pressure_mean;black_foam_temperature_mean] - [car_sponge_pressure_mean;car_sponge_temperature_mean])';
[V,D,U] = eig(Sw\Sb);
V_1 = V(:,1);
plot([-V_1(1),V_1(1)]*2.5,[V_1(2),-V_1(2)]*2.5);
%temperature vs vibration

subplot(1,3,3)
scatter(black_foam_temperature,black_foam_vibration,'filled','black');
hold on;
scatter(car_sponge_temperature,car_sponge_vibration,'filled','yellow');
xlabel('temperature');
ylabel('vibration');
Sw = ([black_foam_temperature;black_foam_vibration] - [black_foam_temperature_mean;black_foam_vibration_mean]) * ([black_foam_temperature;black_foam_vibration] - [black_foam_temperature_mean;black_foam_vibration_mean])' + ([car_sponge_temperature;car_sponge_vibration] - [car_sponge_temperature_mean;car_sponge_vibration_mean]) * ([car_sponge_temperature;car_sponge_vibration] - [car_sponge_temperature_mean;car_sponge_vibration_mean])';
Sb = ([black_foam_temperature_mean;black_foam_vibration_mean] - [car_sponge_temperature_mean;car_sponge_vibration_mean]) * ([black_foam_temperature_mean;black_foam_vibration_mean] - [car_sponge_temperature_mean;car_sponge_vibration_mean])';
[V,D,U] = eig(Sw\Sb);
V_1 = V(:,1);
plot([-V_1(1),V_1(1)]*2.5,[V_1(2),-V_1(2)]*2.5);

%part b
figure;
scatter3(black_foam_pressure,black_foam_vibration,black_foam_temperature,'filled','black');
hold on;
scatter3(car_sponge_pressure,car_sponge_vibration,car_sponge_temperature,'filled','yellow');
xlabel('pressure');
ylabel('vibration');
zlabel('temperature');
% Sw = ([black_foam_pressure;black_foam_vibration;black_foam_temperature] - [black_foam_pressure_mean;black_foam_vibration_mean;black_foam_temperature_mean]) * ([black_foam_pressure;black_foam_vibration;black_foam_temperature] - [black_foam_pressure_mean;black_foam_vibration_mean;black_foam_temperature_mean])' + ([car_sponge_pressure;car_sponge_vibration;car_sponge_temperature] - [car_sponge_pressure_mean;car_sponge_vibration_mean;car_sponge_temperature_mean]) * ([car_sponge_pressure;car_sponge_vibration;car_sponge_temperature] - [car_sponge_pressure_mean;car_sponge_vibration_mean;car_sponge_temperature_mean])';
% Sb = ([black_foam_pressure_mean;black_foam_vibration_mean;black_foam_temperature_mean] - [car_sponge_pressure_mean;car_sponge_vibration_mean;car_sponge_temperature_mean]) * ([black_foam_pressure_mean;black_foam_vibration_mean;black_foam_temperature_mean] - [car_sponge_pressure_mean;car_sponge_vibration_mean;car_sponge_temperature_mean])';

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
black_foam_data = two_data(:,1:10);
car_sponge_data = two_data(:,11:20);


% lambda=diag(D);
% [lambda, SortOrder]=sort(lambda,'descend');
% V=V(:,SortOrder);

black_foam_data = black_foam_data' * V;
car_sponge_data = car_sponge_data' * V;

%plot the projection
figure;
subplot(3,1,1);

scatter(black_foam_data(:,1),0,"black",'*');
hold on;
scatter(mean(black_foam_data(:,1)),0,'black');
scatter(car_sponge_data(:,1),0,"red",'*');
scatter(mean(car_sponge_data(:,1)),0,"red");
title('LDA component 1');

subplot(3,1,2);
scatter(black_foam_data(:,2),0,"black",'*');
hold on;
scatter(mean(black_foam_data(:,2)),0,'black');
scatter(car_sponge_data(:,2),0,"red",'*');
scatter(mean(car_sponge_data(:,2)),0,"red");
title('LDA component 2');

subplot(3,1,3);
scatter(black_foam_data(:,3),0,"black",'*');
hold on;
scatter(mean(black_foam_data(:,3)),0,'black');
scatter(car_sponge_data(:,3),0,"red",'*');
scatter(mean(car_sponge_data(:,3)),0,"red");
title('LDA component 3')