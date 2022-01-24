function [x,y,x_L,y_L,head_angle,angle_s,Tail] = linear_mean_path_v2(x,y,ribbon_x,ribbon_y,...
    angle_s,d_angle_s,Direction,velocity,FPS,Rand_Displace,harmonic_R,temp,temp2,temp3,Tail_length,Tail_cycle_n)


%% LINEAR MEAN

fy = @(angle) -(sind(angle) + harmonic_R*sind(3*angle));
[~,fval] = fminbnd(fy,0,90);

rib_mov(:) = [ribbon_x*-(sind(2*angle_s)) ribbon_y*fy(angle_s)/(-fval)];
rib_mov_ribbon(:) = [ribbon_x*-(sind(2*angle_s)) -sind(angle_s)*ribbon_y];
delta = rib_mov - rib_mov_ribbon;

%Rotation
TF = [cosd(Direction) -sind(Direction); sind(Direction) cosd(Direction)];
rot_rib(:) = TF*rib_mov(:);

% Calculation of head angle
rib_mov_1(:) = [ribbon_x*-sind(2*(angle_s+0.001)); ribbon_y*fy(angle_s+0.001)/(-fval)];
rot_rib_1(:) = TF*rib_mov_1(:);
head_angle = atan2d((rot_rib_1(2) - rot_rib(2)),(rot_rib_1(1) - rot_rib(1)));

x_L = x + rot_rib(1);
y_L = y + rot_rib(2);

Tail = LM_Tail(x_L,y_L,ribbon_x,ribbon_y,angle_s,Direction,temp,temp2,temp3,Tail_length,Tail_cycle_n,delta,TF);

% Update the positions of all the particles
angle_s = angle_s + 360*d_angle_s/FPS;
x = x + velocity* cosd(Direction)/FPS + Rand_Displace*randn/sqrt(FPS);
y = y + velocity* sind(Direction)/FPS + Rand_Displace*randn/sqrt(FPS);

% %Tail model
% angle_s_Tail = angle_s;
% x_Tail = 0;
% y_Tail = 0; 
% 
% 
% for M = 1:length(temp)
% 
%     rib_mov(:) = [ribbon_x*-(sind(2*angle_s_Tail)) ribbon_y*-sind(angle_s_Tail)];
% 
%     rib_mov(1) = temp(M)*rib_mov(1);
%     rib_mov(2) = temp2(M)*rib_mov(2);
%     rib_mov(2) = rib_mov(2) - temp3(M)*delta(2);
% 
%     %Rotation
%     rot_rib(:) = TF*rib_mov(:);
% 
%     x_Tail_L(M) = x_Tail + rot_rib(1);
%     y_Tail_L(M) = y_Tail + rot_rib(2);
% 
%     % Update the positions of all the particles
%     angle_s_Tail = angle_s_Tail - 360*Tail_cycle_n/length(temp);
%     x_Tail = x_Tail - Tail_length * cosd(Direction)/length(temp);
%     y_Tail = y_Tail - Tail_length * sind(Direction)/length(temp);           
% 
%     Tail = [x_Tail_L - x_Tail_L(1); y_Tail_L - y_Tail_L(1)];
% 
% end