function Tail = C_Tail(x_L,y_L,ampli,beat_f,k,FPS,angle_cs,angle_c,d_angle_c,b,L_T,lambda)

y0_T =  b(0) * ampli * -sind(-360*(beat_f*k/FPS) - angle_cs);
y_T = (b(L_T).* ampli .* -sind(360*(L_T/lambda - beat_f*k/FPS) - angle_cs)) - y0_T;
TF = [cosd(angle_c-90) -sind(angle_c-90); sind(angle_c-90) cosd(angle_c-90)];

if d_angle_c>0
    Tail = [L_T; y_T];
else
    Tail = [-L_T; y_T];
end

for n = 1:length(y_T)
    Tail(:,n) = TF*Tail(:,n);
end

Tail(1,:) = Tail(1,:) + x_L;
Tail(2,:) = Tail(2,:) + y_L;