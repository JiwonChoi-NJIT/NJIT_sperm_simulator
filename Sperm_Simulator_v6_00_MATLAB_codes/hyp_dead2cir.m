function [distribution,angle_c,radius,d_angle_c,ampli,beat_f,angle_cs,center_r_x,center_r_y] = ...
          hyp_dead2cir(radius_v,radius_m,var_d,ave_d,ampli_v,ampli_m,beat_v,beat_m,k,FPS,x_L,y_L)


% Change swim type from hyperactive to circular
distribution = 1;


%starting angles for the particle
angle_c = rand*360;


%Parameter Resets
radius =        radius_v*randn          + radius_m;
d_angle_c =     var_d*abs(randn)        + ave_d;
ampli =         ampli_v*randn           + ampli_m;
beat_f =        abs(beat_v*randn        + beat_m);
angle_cs =      180*rand;


if rand < 0.5
    d_angle_c   = -d_angle_c;
end


x_temp = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs)) * cosd(angle_c);
y_temp = (radius + ampli*sind(360*beat_f*k/FPS + angle_cs)) * sind(angle_c);


center_r_x = x_L - x_temp;
center_r_y = y_L - y_temp;