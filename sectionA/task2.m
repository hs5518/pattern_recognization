clc;
clear;
time_step = 350;
mat_acrylic = ["acrylic_211_01_HOLD", "acrylic_211_02_HOLD","acrylic_211_03_HOLD","acrylic_211_04_HOLD","acrylic_211_05_HOLD","acrylic_211_06_HOLD","acrylic_211_07_HOLD","acrylic_211_08_HOLD","acrylic_211_09_HOLD","acrylic_211_10_HOLD"];
mat_black_foam = ["black_foam_110_01_HOLD", "black_foam_110_02_HOLD","black_foam_110_03_HOLD","black_foam_110_04_HOLD","black_foam_110_05_HOLD","black_foam_110_06_HOLD","black_foam_110_07_HOLD","black_foam_110_08_HOLD","black_foam_110_09_HOLD","black_foam_110_10_HOLD"];
mat_car_sponge = ["car_sponge_101_01_HOLD","car_sponge_101_02_HOLD","car_sponge_101_03_HOLD","car_sponge_101_04_HOLD","car_sponge_101_05_HOLD","car_sponge_101_06_HOLD","car_sponge_101_07_HOLD","car_sponge_101_08_HOLD","car_sponge_101_09_HOLD","car_sponge_101_10_HOLD"];
mat_flour_sack = ["flour_sack_410_01_HOLD","flour_sack_410_02_HOLD","flour_sack_410_03_HOLD","flour_sack_410_04_HOLD","flour_sack_410_05_HOLD","flour_sack_410_06_HOLD","flour_sack_410_07_HOLD","flour_sack_410_08_HOLD","flour_sack_410_09_HOLD","flour_sack_410_10_HOLD"];
mat_kitchen_sponge = ["kitchen_sponge_114_01_HOLD","kitchen_sponge_114_02_HOLD","kitchen_sponge_114_03_HOLD","kitchen_sponge_114_04_HOLD","kitchen_sponge_114_05_HOLD","kitchen_sponge_114_06_HOLD","kitchen_sponge_114_07_HOLD","kitchen_sponge_114_08_HOLD","kitchen_sponge_114_09_HOLD","kitchen_sponge_114_10_HOLD"];
mat_steel_vase = ["steel_vase_702_01_HOLD", "steel_vase_702_02_HOLD", "steel_vase_702_03_HOLD", "steel_vase_702_04_HOLD", "steel_vase_702_05_HOLD", "steel_vase_702_06_HOLD", "steel_vase_702_07_HOLD", "steel_vase_702_08_HOLD", "steel_vase_702_09_HOLD", "steel_vase_702_10_HOLD"];
prefix = "D:\scripts\pattern recognization\PR_CW_DATA_2021\";
%pressure,vibration,temperature
acrylic = zeros(3,10);
black_foam = zeros(3,10);
car_sponge = zeros(3,10);
flour_sack = zeros(3,10);
kitchen_sponge = zeros(3,10);
steel_vase = zeros(3,10);

for i = 1:length(mat_acrylic)
    load(prefix+mat_acrylic(i))
    acrylic(1,i) = F0pdc(time_step);
    acrylic(2,i) = F0pac(2,time_step);
    acrylic(3,i) = F0tac(time_step);
end

for i = 1:length(black_foam)
    load(prefix+mat_black_foam(i))
    black_foam(1,i) = F0pdc(time_step);
    black_foam(2,i) = F0pac(2,time_step);
    black_foam(3,i) = F0tac(time_step);
end

for i = 1:length(car_sponge)
    load(prefix+mat_car_sponge(i))
    car_sponge(1,i) = F0pdc(time_step);
    car_sponge(2,i) = F0pac(2,time_step);
    car_sponge(3,i) = F0tac(time_step);
end

for i = 1:length(flour_sack)
    load(prefix+mat_flour_sack(i))
    flour_sack(1,i) = F0pdc(time_step);
    flour_sack(2,i) = F0pac(2,time_step);
    flour_sack(3,i) = F0tac(time_step);
end

for i = 1:length(kitchen_sponge)
    load(prefix+mat_kitchen_sponge(i))
    kitchen_sponge(1,i) = F0pdc(time_step);
    kitchen_sponge(2,i) = F0pac(2,time_step);
    kitchen_sponge(3,i) = F0tac(time_step);
end

for i = 1:length(steel_vase)
    load(prefix+mat_steel_vase(i))
    steel_vase(1,i) = F0pdc(time_step);
    steel_vase(2,i) = F0pac(2,time_step);
    steel_vase(3,i) = F0tac(time_step);
end

save('F0_PVT','acrylic','black_foam','car_sponge','flour_sack','kitchen_sponge','steel_vase');
