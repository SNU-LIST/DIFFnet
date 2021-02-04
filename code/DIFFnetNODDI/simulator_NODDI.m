function[S_matrix,gradient_matrix,info_matrix,signal_num_matrix] = simulator_NODDI(iter)

%
%  Description:
%  Monte-Carlo diffusion simulation for NODDI model
%
%  Copyright @ Juhyung Park
%  Laboratory for Imaging Science and Technology
%  Seoul National University
%  email : jack0878@snu.ac.kr
%

S_matrix = zeros([iter 160]);
gradient_matrix = zeros([iter 160 3]);
info_matrix = zeros([iter 3]);
signal_num_matrix = zeros([iter 3]);

gyroratio = 2*pi*42.57 * 10^3;
Smalldelta = 0.033;
Largedelta = 0.046;
timestep = 0.001;
timenum = (Smalldelta+Largedelta)/timestep;
Gstep = zeros([1 timenum]);
Gstep(1:(Smalldelta/timestep)) = 1;
Gstep((Largedelta/timestep):((Smalldelta+Largedelta)/timestep)) = -1;
Gmax = sqrt((2300*1e+6)/(Largedelta-Smalldelta/3))/(gyroratio*Smalldelta);
voxel_length = 2*1e-3;                                                     
seed_num = 6000;
proton_num = 1e+6;

fprintf('\nStart Diffusion simulation\n')
fprintf('Proton number : %d\n' , proton_num)
fprintf('    Iter       Viso     Vic      OD      time taken(s)\n')

for iteration = 1:iter
starttime = tic(); 

signal_num1=randi(5)+5;
signal_num2=randi(25)+25;
signal_num3=randi(50)+50;
signal_num = [signal_num1 signal_num2 signal_num3];
signal_num_sum = sum(signal_num);

b1 = (200*rand()+200)*1e+6;
b2 = (400*rand()+500)*1e+6;
b3 = (600*rand()+1700)*1e+6;
b = [b1 b2 b3];
G = sqrt(b/(Largedelta-Smalldelta/3))/(gyroratio*Smalldelta);

locseed = rand(seed_num, 4)*2-1;
locseed(:,4) = sqrt(locseed(:,1).^2+locseed(:,2).^2+locseed(:,3).^2);
locseed(:,1:3) = locseed(:,1:3)./repmat(locseed(:,4),1,3);
randvector = [];

for seed = 1:seed_num
if locseed(seed,4) < 1
    randvector = [randvector; locseed(seed,:)];
end
end

gradientb1 = randvector(1:signal_num(1),1:3);
gradientb2 = randvector(signal_num(1)+1:signal_num(1)+signal_num(2),1:3);
gradientb3 = randvector(signal_num(2)+1:signal_num(2)+signal_num(3),1:3);  

cart = randvector(signal_num(2)+signal_num(3)+1,1:3);
lpi = atan(cart(2)/cart(1));

if cos(lpi) == 0
    theta = atan(cart(2)/sin(lpi)/cart(3));
else
    theta = atan(cart(1)/cos(lpi)/cart(3));
    if isnan(theta)
        theta = atan(cart(2)/sin(lpi)/cart(3));
    end
end
    
SNR =  30+rand()*70;
noise_level = 1/SNR;

viso = rand();
vic = rand();
od =rand()*0.998 + 0.001;

k=tan(od*pi/2)^-1;
f_k = 0.5 * erfi(sqrt(k))*exp(-k)*sqrt(pi);
tau = -1/(2*k) + 1/(2*f_k*sqrt(k));

viso_proton_num = round(proton_num * viso);
vic_proton_num = round((proton_num - viso_proton_num)*vic);
vex_proton_num = proton_num - vic_proton_num - viso_proton_num;
 
proton_pos = (((1/1e6) * randi(1e6,1,3*proton_num) - (1/2))*voxel_length);
proton_pos = reshape(proton_pos, round(proton_num*1), 3);    

D_coeff = [3 3 3;...
           1.7*(1-vic*((1+tau)/2)) 1.7*(1-vic*((1+tau)/2)) 1.7*(1-vic*(1-tau));...
           0 0 1.7;]*1e-9;     
        
step_std = sqrt(2*D_coeff*timestep);

axis_D1 = [ 1, 0, 0;...
            0, 1, 0;...
            0, 0, 1];                                

axis_D2 = [cos(theta)*cos(lpi),    cos(theta)*sin(lpi),    -sin(theta);...                             
           -sin(lpi),              cos(lpi),               0;...                              
           sin(theta)*cos(lpi),    sin(theta)*sin(lpi),    cos(theta)];

                               
axonseed = randvector(signal_num(2)+signal_num(3)+2:signal_num(2)+signal_num(3)+501,1:3);
u = [sin(theta)*cos(lpi) sin(theta)*sin(lpi) cos(theta)]';
h = hypergeom(1/2,3/2,k)^-1;
f_n =  h* exp(k*((axonseed*u).^2));
len = length(f_n);
f_n_cumsum = cumsum(f_n);
maxv = f_n_cumsum(500);
rand_point = rand(vic_proton_num,1)*maxv*0.8 + maxv*0.1;
difference_matrix = abs(repmat(f_n_cumsum',vic_proton_num,1) - repmat(rand_point,1,len));
[~,a2] = min(difference_matrix,[],2);
axis_D3 =axonseed(a2,:);

signal = zeros([proton_num 3]);

for  t = 1:timenum
        move_step1 = [  normrnd(0,step_std(1,1), [viso_proton_num 1])...
                        normrnd(0,step_std(1,2), [viso_proton_num 1])...
                        normrnd(0,step_std(1,3), [viso_proton_num 1])];
        move_step2 = [  normrnd(0,step_std(2,1), [vex_proton_num 1])...
                        normrnd(0,step_std(2,2), [vex_proton_num 1])...
                        normrnd(0,step_std(2,3), [vex_proton_num 1])];
        move_step3 = normrnd(0,step_std(3,3), [vic_proton_num 1]) ;
        move_step3_3 = repmat(move_step3,1,3);
        proton_pos = proton_pos + [move_step1* axis_D1; move_step2* axis_D2;  move_step3_3.*axis_D3];
        signal = signal +  Gstep(t)*gyroratio*timestep.*(proton_pos);
end

S1 = (mean(exp(-1j * signal * G(1) * gradientb1'),1));
S2 = (mean(exp(-1j * signal * G(2) * gradientb2'),1));
S3 = (mean(exp(-1j * signal * G(3) * gradientb3'),1));
S = [S1 S2 S3];
S = abs(S + normrnd(0,noise_level,[1 signal_num_sum]) + 1j*normrnd(0,noise_level,[1 signal_num_sum]));

S_matrix(iteration,1:signal_num_sum) = S;
gradient_matrix(iteration,1:signal_num_sum,:) = [gradientb1*G(1); gradientb2*G(2); gradientb3*G(3)]/Gmax;
info_matrix(iteration,:) = [viso vic od];
signal_num_matrix(iteration,:) = signal_num;

time_taken = toc(starttime);

if mod(iteration,10) == 0
fprintf('%6d  ',iteration)
fprintf('        %0.4f,  %0.4f,  %0.4f    ',viso,vic,od);
fprintf('                  %0.3f \n', time_taken)
end

end
end