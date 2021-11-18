function [center_r_x,center_r_y,x,y,x_L,y_L,head_angle,angle_c,Tail] = circular_path_v2(x,y,radius,ampli,...
            beat_f,k,FPS,angle_cs,angle_c,d_angle_c,center_r_x,center_r_y,b,L_T,lambda,Rand_Displace)


%% CIRCULAR

center_r_x = center_r_x +  Rand_Displace*randn/sqrt(FPS);
center_r_y = center_r_y +  Rand_Displace*randn/sqrt(FPS);

x_L = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs))...
            *cosd(angle_c) + center_r_x;
y_L = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs))...
            *sind(angle_c) + center_r_y;

x_L_1 = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs + beat_f*360/1000))...
            *cosd(angle_c+d_angle_c/1000) + center_r_x;

y_L_1 = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs + beat_f*360/1000))...
            *sind(angle_c+d_angle_c/1000) + center_r_y;        

% y0_T =  b(0) * ampli * -sind(-360*(beat_f*k/FPS) - angle_cs);
% y_T = (b(L_T).* ampli .* -sind(360*(L_T/lambda - beat_f*k/FPS) - angle_cs)) - y0_T;
% TF = [cosd(angle_c-90) -sind(angle_c-90); sind(angle_c-90) cosd(angle_c-90)];
% 
% if d_angle_c>0
%     Tail = [L_T; y_T];
% else
%     Tail = [-L_T; y_T];
% end
% 
% for n = 1:length(y_T)
%     Tail(:,n) = TF*Tail(:,n);
% end

Tail = C_Tail(x_L,y_L,ampli,beat_f,k,FPS,angle_cs,angle_c,d_angle_c,b,L_T,lambda);

head_angle = atan2d((y_L_1 - y_L),(x_L_1 - x_L));
angle_c = angle_c + d_angle_c/FPS;   % Scale to match desired FPS



