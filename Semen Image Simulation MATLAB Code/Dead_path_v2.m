function [x,y,x_L,y_L,head_angle,Tail] = Dead_path_v2(x,y,head_angle,Hyper_Disp,Tail,L_T,FPS,Rand_Displace)

x_random = Rand_Displace*randn/sqrt(FPS);
y_random = Rand_Displace*randn/sqrt(FPS);

x_L = x + x_random;
y_L = y + y_random;

x = x_L;
y = y_L;

if size(Tail,2) == 0
    head_angle = rand*360;
        
    Tail = cumsum(Hyper_Disp*sqrt(1/(ceil(FPS)))*randn(2,ceil(FPS/5)),2) + [x y]';
    L_map = linspace(0,1,ceil(FPS/5));
    L_interpolated = linspace(0,1,length(L_T));
    X_interpolated = interp1(L_map,Tail(1,:),L_interpolated);
    Y_interpolated = interp1(L_map,Tail(2,:),L_interpolated);
    Tail = [X_interpolated;Y_interpolated];
else
    Tail(1,:) = Tail(1,:) + x_random;
    Tail(2,:) = Tail(2,:) + y_random;
end
