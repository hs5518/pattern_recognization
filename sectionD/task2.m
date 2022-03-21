close all;
clear;
clc;

time_step = 313;
mat_name = ["acrylic_211_01_HOLD", "acrylic_211_02_HOLD","acrylic_211_03_HOLD","acrylic_211_04_HOLD","acrylic_211_05_HOLD","acrylic_211_06_HOLD","acrylic_211_07_HOLD","acrylic_211_08_HOLD","acrylic_211_09_HOLD","acrylic_211_10_HOLD","black_foam_110_01_HOLD", "black_foam_110_02_HOLD","black_foam_110_03_HOLD","black_foam_110_04_HOLD","black_foam_110_05_HOLD","black_foam_110_06_HOLD","black_foam_110_07_HOLD","black_foam_110_08_HOLD","black_foam_110_09_HOLD","black_foam_110_10_HOLD","car_sponge_101_01_HOLD","car_sponge_101_02_HOLD","car_sponge_101_03_HOLD","car_sponge_101_04_HOLD","car_sponge_101_05_HOLD","car_sponge_101_06_HOLD","car_sponge_101_07_HOLD","car_sponge_101_08_HOLD","car_sponge_101_09_HOLD","car_sponge_101_10_HOLD","flour_sack_410_01_HOLD","flour_sack_410_02_HOLD","flour_sack_410_03_HOLD","flour_sack_410_04_HOLD","flour_sack_410_05_HOLD","flour_sack_410_06_HOLD","flour_sack_410_07_HOLD","flour_sack_410_08_HOLD","flour_sack_410_09_HOLD","flour_sack_410_10_HOLD","kitchen_sponge_114_01_HOLD","kitchen_sponge_114_02_HOLD","kitchen_sponge_114_03_HOLD","kitchen_sponge_114_04_HOLD","kitchen_sponge_114_05_HOLD","kitchen_sponge_114_06_HOLD","kitchen_sponge_114_07_HOLD","kitchen_sponge_114_08_HOLD","kitchen_sponge_114_09_HOLD","kitchen_sponge_114_10_HOLD","steel_vase_702_01_HOLD", "steel_vase_702_02_HOLD", "steel_vase_702_03_HOLD", "steel_vase_702_04_HOLD", "steel_vase_702_05_HOLD", "steel_vase_702_06_HOLD", "steel_vase_702_07_HOLD", "steel_vase_702_08_HOLD", "steel_vase_702_09_HOLD", "steel_vase_702_10_HOLD"];
prefix = "C:\Users\Sherl\Desktop\PR_CW_DATA_2021\";
total_data = [];
for i = 1:length(mat_name)
    load(prefix+mat_name(i));
    total_data = [total_data F0Electrodes(:,time_step)];
end

% Creat the train and test set. (train: 60%, test: 40%)
pointSet=total_data';
cv = cvpartition(size(pointSet,1),'HoldOut',0.4);
idx = cv.test;
% Separate the data to training and test data
dataTrain1 = pointSet(~idx,:);
dataTest1  = pointSet(idx,:);
% label all data:
class_name=["acrylic","black foam","car sponge","flour sack","kitchen sponge","steel vase"];
label_matrix="temp";
for i=1:size(total_data,2)
    for j=1:size(total_data,1)
        %label_matrix(i,j)=class_name(floor((i-1)/10)+1);
        label_matrix(j,i)=class_name(floor((i-1)/10)+1)
    end
end
labelset=label_matrix';
dataTrainlabel=labelset(~idx,:);
dataTestlabel=labelset(idx,:);
%renaming the variables;
DTrain=dataTrain1;
DTest=dataTest1;
DTrainL=dataTrainlabel;
DTestL=dataTestlabel;


Mdl = TreeBagger(50,DTrain,DTrainL,'OOBPrediction','On','Method','classification')
view(Mdl.Trees{1},'Mode','graph')
predicted_labels = predict(Mdl,DTest);

figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Trees number';
ylabel 'Out-of-bag error';

