function [x,y,x_L,y_L,head_angle,Tail] = hyperactive_path_v2(x,y,Hyper_Disp,Rand_Displace,Tail,L_T,FPS)

x_random = Rand_Displace*randn/sqrt(FPS);
y_random = Rand_Displace*randn/sqrt(FPS);

x_L = x + Hyper_Disp*randn/sqrt(FPS) + x_random;
y_L = y + Hyper_Disp*randn/sqrt(FPS) + y_random;
head_angle = atan2d((y_L - y),(x_L - x));
x = x_L;
y = y_L;



if size(Tail,2) == 0
    Tail = cumsum(Hyper_Disp*sqrt(1/(ceil(FPS)))*randn(2,ceil(FPS/5)),2) + [x y]';
    L_map = linspace(0,1,ceil(FPS/5));
    L_interpolated = linspace(0,1,length(L_T));
    X_interpolated = interp1(L_map,Tail(1,:),L_interpolated);
    Y_interpolated = interp1(L_map,Tail(2,:),L_interpolated);
    Tail = [X_interpolated;Y_interpolated];
else
    Tail(1,:) = Tail(1,:) + x_random;
    Tail(2,:) = Tail(2,:) + y_random;
    
    L_keep = ceil(size(Tail,2)*(1-1/(0.2*FPS)));
    L_map = length(L_T) - L_keep;
    L_interpolated = linspace(0,1,L_map);
    X_interpolated = interp1([0 1],[x Tail(1,1)],L_interpolated);
    Y_interpolated = interp1([0 1],[y Tail(2,1)],L_interpolated);
    Tail = [[X_interpolated;Y_interpolated] Tail(:,1:L_keep)];
    
end