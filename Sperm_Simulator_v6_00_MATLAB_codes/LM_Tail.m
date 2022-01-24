function Tail = LM_Tail(x_L,y_L,ribbon_x,ribbon_y,...
    angle_s,Direction,temp,temp2,temp3,Tail_length,Tail_cycle_n,delta,TF)

%Tail model
angle_s_Tail = angle_s;
x_Tail = 0;
y_Tail = 0; 

for M = 1:length(temp)

    rib_mov(:) = [ribbon_x*-(sind(2*angle_s_Tail)) ribbon_y*-sind(angle_s_Tail)];

    rib_mov(1) = temp(M)*rib_mov(1);
    rib_mov(2) = temp2(M)*rib_mov(2);
    rib_mov(2) = rib_mov(2) - temp3(M)*delta(2);

    %Rotation
    rot_rib(:) = TF*rib_mov(:);

    x_Tail_L(M) = x_Tail + rot_rib(1);
    y_Tail_L(M) = y_Tail + rot_rib(2);

    % Update the positions of all the particles
    angle_s_Tail = angle_s_Tail - 360*Tail_cycle_n/length(temp);
    x_Tail = x_Tail - Tail_length * cosd(Direction)/length(temp);
    y_Tail = y_Tail - Tail_length * sind(Direction)/length(temp);           

    Tail = [x_Tail_L - x_Tail_L(1); y_Tail_L - y_Tail_L(1)];

end

Tail(1,:) = Tail(1,:) + x_L;
Tail(2,:) = Tail(2,:) + y_L;
