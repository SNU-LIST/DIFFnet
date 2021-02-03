function[S_matrix,gradient_matrix,info_matrix,signal_num_matrix] = DTIsimulation(proton_num, iter)

S_matrix = zeros([iter 80]);
gradient_matrix = zeros([iter 80 3]);
info_matrix = zeros([iter 4]);
signal_num_matrix = zeros([iter 1]);

gyroratio = 2*pi*42.57 * 10^3;
Smalldelta = 0.033;
Largedelta = 0.046;
timestep = 0.001;
lobenum = Smalldelta/timestep;
timenum = lobenum*2+1;
Gstep = zeros([1 timenum]);
Gstep(1:lobenum) = 1;
Gstep(lobenum+2:lobenum*2+1) = -1;
Gmax = sqrt((350*1e+6)/(Largedelta-Smalldelta/3))/(gyroratio*Smalldelta);
voxel_length = 2*1e-3;  

fprintf('\nStart Diffusion simulation\n')
fprintf('Proton number : %d\n' , proton_num)
fprintf('Iter             FA       MD       AD       RD            time taken(s)\n')

for iteration = 1:iter
starttime = tic();

signal_num=randi(60)+20;

b = (600*rand()+600)*1e+6;
G = sqrt(b/(Largedelta-Smalldelta/3))/(gyroratio*Smalldelta);

loc = rand(500, 4)*2-1;
loc(:,4) = sqrt(loc(:,1).^2+loc(:,2).^2+loc(:,3).^2);
loc(:,1:3) = loc(:,1:3)./repmat(loc(:,4),1,3);
randvector = [];
for ii = 1:500
if loc(ii,4) < 1
    randvector = [randvector; loc(ii,:)];
end
end
gradient = randvector(1:signal_num,1:3);
cart = randvector(signal_num+1,1:3);
lpi = atan(cart(2)/cart(1));
if cos(lpi) == 0
    theta = atan(cart(2)/sin(lpi)/cart(3));
else
    theta = atan(cart(1)/cos(lpi)/cart(3));
    if isnan(theta)
        theta = atan(cart(2)/sin(lpi)/cart(3));
    end
end

axis_D1 = [cos(theta)*cos(lpi),    cos(theta)*sin(lpi),     -sin(theta);...                             
                     -sin(lpi),               cos(lpi),               0;...                              
           sin(theta)*cos(lpi),    sin(theta)*sin(lpi),      cos(theta)];

SNR = 60+rand() * 100;
noise_level = 1/SNR;

L1 = rand();
L2 = rand()*L1;
L3 = rand()*L2;

D_coeff1 = [L1 L2 L3]*3.5e-9;  

step_std1 = sqrt(2*D_coeff1*timestep);
step_std_zero = sqrt(2*D_coeff1*timestep*13);

proton_pos = (((1/1e6) * randi(1e6,1,3*proton_num) - (1/2))*voxel_length);
proton_pos = reshape(proton_pos, round(proton_num*1), 3);   
signal = zeros([proton_num 3]);

for  t = 1:timenum
    if Gstep(t) == 0
        move_step1 = [normrnd(0,step_std_zero(1), [proton_num 1])...
                  normrnd(0,step_std_zero(2), [proton_num 1])...
                  normrnd(0,step_std_zero(3), [proton_num 1])];
        proton_pos(:,1:3) = proton_pos(:,1:3) + move_step1* axis_D1 ;
    else
        move_step1 = [normrnd(0,step_std1(1), [proton_num 1])...
                      normrnd(0,step_std1(2), [proton_num 1])...
                      normrnd(0,step_std1(3), [proton_num 1])];
        proton_pos(:,1:3) = proton_pos(:,1:3) + move_step1* axis_D1 ;
        signal = signal + G * Gstep(t)*gyroratio*timestep.*(proton_pos);
    end
end

S = (mean(exp(-1j * signal * gradient'),1));
S = abs(S + normrnd(0,noise_level,[1 signal_num]) + 1j*normrnd(0,noise_level,[1 signal_num]));

FA = sqrt(((L1-L2)^2 + (L2-L3)^2 + (L1-L3)^2))/sqrt(2*(L1^2+L2^2 + L3^2));
MD = (L1+L2+L3)/3;
AD = L1;
RD = (L2+L3)/2;

S_matrix(iteration,1:signal_num) = S;
gradient_matrix(iteration,1:signal_num,:) = gradient * G / Gmax;
info_matrix(iteration,:) = [FA MD AD RD];
signal_num_matrix(iteration) = signal_num;

time_taken = toc(starttime);

if mod(iteration,10) == 0
fprintf('%6d  ',iteration)
fprintf('        %0.4f,  %0.4f,  %0.4f   %0.4f ',FA,MD,AD,RD);
fprintf('       %0.3f \n', time_taken)      
end


end
end