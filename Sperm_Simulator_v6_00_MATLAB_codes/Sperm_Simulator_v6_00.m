function varargout = Sperm_Simulator_v6_00(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Sperm_Simulator_v6_00_OpeningFcn, ...
                   'gui_OutputFcn',  @Sperm_Simulator_v6_00_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function Sperm_Simulator_v6_00_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

guidata(hObject, handles);


function varargout = Sperm_Simulator_v6_00_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;

%%
% Changes made June 20th 2019 (Version 3)
% 1. Circular path equation incorrect (Frequency modulated on circular path
% was period/rotation. Now correctly fixed to period/second)
% 2. All the functions did not correctly reflect on the frame rate. Now the
% changes are correct according to the framerate specified by the user.

% Changes made July 30th 2019 (Version 4)


% --- Executes on button press in Sim_start.
function Sim_start_Callback(hObject, eventdata, handles)

if get(handles.RNG_setting,'value') == 1
    rng(str2double(get(handles.RNG_index,'String')))
else
    rng shuffle
end

% VERY IMPORTANT PARAMETERS!!!!!
numPixels = str2double(get(handles.Frame_size,'String'));
FPS = str2double(get(handles.FPS,'String'));
numFrames = str2double(get(handles.Video_time,'String'))*FPS;


%% Special Cell Density
% Number of particles
numParticles = str2double(get(handles.particle_n,'String'));

Dead = str2double(get(handles.Dead_P,'String'))/100;
if Dead < 0 || Dead > 1
   errordlg('Please Check the distribution for dead cells','Swim Distribution Error')
   return
end

Hyper = str2double(get(handles.Hyper_P,'String'))/100;
if Hyper < 0 || Hyper > 1
   errordlg('Please Check the distribution for hyperactive cells','Swim Distribution Error')
   return
end

str_d = str2double(get(handles.Straight_Dis,'String'))/100;
if str_d < 0 || str_d > 1
   errordlg('Please Check the distribution for linear mean swimming cells','Swim Distribution Error')
   return
end

% Number of Dead/Weak Cells;
DeadSperm = round(Dead*numParticles);
numParticle_HLC = numParticles - DeadSperm;

if numParticle_HLC ~= 0
    temp_percent = 1-Dead;
    HyperSperm = round((Hyper/temp_percent)*numParticle_HLC);
    numParticle_LC = numParticle_HLC - HyperSperm;

    if numParticle_LC ~= 0
        temp_percent = temp_percent - Hyper;
        LinearMeanSperm = round((str_d/temp_percent)*numParticle_LC);
    else
        LinearMeanSperm = 0;
    end
else
    HyperSperm = 0;
    LinearMeanSperm = 0;
end
if numParticles < DeadSperm + HyperSperm + LinearMeanSperm
   errordlg('Swim Type Distribution Set Incorrectly: Over 100 Percent Sum','Swim Distribution Error')
   return
end
%% Circular motion sperm
%particle beating frequency/amplitude
beat_v = str2double(get(handles.Beat_v,'String'));
beat_m = str2double(get(handles.Beat_m,'String'));
ampli_v = str2double(get(handles.Amp_v,'String'));
ampli_m = str2double(get(handles.Amp_m,'String'));

beat_f = abs(beat_v*(randn(1,numParticles))+beat_m);
ampli = ampli_v*randn(1,numParticles)+ampli_m;

%radius of each particle spin
radius_m = str2double(get(handles.Radius_m,'String'));
radius_v = str2double(get(handles.Radius_v,'String'));
radius = radius_v*(randn(1,numParticles))+radius_m;

%average change of angle and its variance
ave_d = str2double(get(handles.AV_m,'String'));
var_d = str2double(get(handles.AV_v,'String'));

%starting angles for each particles
angle_c = 360*rand(1,numParticles);

%Determination of LCh or RCh
for i = 1: numParticles
    if rand > 0.5
        d_angle_c(i) = var_d*abs(randn(1,1))+ave_d;
    else
        d_angle_c(i) = -(var_d*abs(randn(1,1))+ave_d);
    end
end

angle_cs = 360*rand(1,numParticles);

%% Linear Mean Swimming Sperm
% Desired linear velocity
variance_LV = str2double(get(handles.LV_v,'String'));
desired_LV = str2double(get(handles.LV_m,'String'));
velocity = abs((variance_LV^.5*randn(1,numParticles))+desired_LV);

% Desired size of ribbon
variance_rib_x = str2double(get(handles.height_v,'String'))/4;
desired_rib_x = str2double(get(handles.height_m,'String'))/2;
ribbon_x = (variance_rib_x^.5*randn(1,numParticles))+desired_rib_x;

variance_rib_y = str2double(get(handles.width_v,'String'))/4;
desired_rib_y = str2double(get(handles.width_m,'String'))/2;
ribbon_y = (variance_rib_y^.5*randn(1,numParticles))+desired_rib_y;

% Desired Angle change rate
variance_A = str2double(get(handles.LV_a_v,'String'));
desired_A = str2double(get(handles.LV_a_m,'String'));
d_angle_s = abs((variance_A^.5*randn(1,numParticles))+desired_A);

% starting angle
angle_s = 360*rand(1,numParticles);

% starting direction distribution (degrees)
Direction = 360*rand(1,numParticles);

Hyper_Disp = str2double(get(handles.Hyper_Disp,'String'));
Rand_Displace = str2double(get(handles.Rand_Displace,'String'));

harmonic_R = str2double(get(handles.harmonic_R,'String'));
%%
V = str2double(get(handles.Vertical_Size,'String'));
H = str2double(get(handles.Horizontal_Size,'String'));
Halo_Ratio = str2double(get(handles.Halo_Ratio,'String'));

Int_H = str2double(get(handles.Intensity_percent_H,'String'));
Int_L = str2double(get(handles.Intensity_percent_L,'String'));
Halo_Int_H = str2double(get(handles.Halo_Intensity,'String'))/100;

I_High = 1 - str2double(get(handles.Background_I,'String'))/100;
I_Low = 1 - str2double(get(handles.Background_I_L,'String'))/100;
%% Tail parameters
Tail_L = str2double(get(handles.Tail_L,'String'));
tail_int  = str2double(get(handles.Tail_Int,'String'));
Tail_width = str2double(get(handles.Tail_W_D,'String'));

% CIRCULAR TAIL PARAMETERS

b = @(x) 0.02*x + 0.8;
lambda = Tail_L;

L_T = linspace(0,lambda,300);

% LINEAR MEAN TAIL PARAMETERS

temp = linspace(-2,20,length(L_T));
temp = 1./(1+exp(temp));
temp = temp/max(temp);

temp2 = linspace(0,1,length(L_T));
temp2 = exp(-1.5*temp2);

temp3 = linspace(0,1,length(L_T));
temp3 = 1 - exp(-5*temp3);

Tail_length = Tail_L*(5/5);
Tail_cycle_n = 1;


%% Set filter (point spread function for head shape)
mu = [0 0];
sigma = [H 0;0 V];

if and(H<5,V<5)
    x1 = -12:12;
    x2 = -12:12;
elseif and(H<10,V<10)
    x1 = -25:25;
    x2 = -25:25;
else
    x1 = -100:100;
    x2 = -100:100;
end
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
    
psf1 = mvnpdf(X,mu,sigma);
psf1 = reshape(psf1,length(x2),length(x1));
psf1 = psf1/sum(psf1,'all');

psf2 = mvnpdf(X,mu,sigma*Halo_Ratio);
psf2 = reshape(psf2,length(x2),length(x1));
psf2 = psf2/sum(psf2,'all');
psf2 = del2(psf2);

psf3 = fspecial('log', 50, Tail_width);
% psf3 = psf3/sum(psf3,'all');
%% Minimum/maximum particle intensity
Imin = 1/max(psf1,[],'all')*Int_L/100;
Imax = 1/max(psf1,[],'all')*Int_H/100;

% Image noise 
sig_Io = str2double(get(handles.Noise_Level,'String'));

% Random particle intensities (uniform over [Imin, Imax])
Io1 = Imin + (Imax - Imin) * rand(1,numParticles);
Io2 = Io1;


%%
% Random initial particle positions (assumed uniform in the FOV)
% (note: this vector summarizes the positions of *all* the particles, it
% has dimension 2 x numParticles i.e., [xpos, ypos] x numParticles)
x = rand(1,numParticles) * numPixels;
y = rand(1,numParticles) * numPixels;

%calculation of each particle's center for spin(for spinning cells)
center_r_x = x - radius.*cosd(angle_c);
center_r_y = y - radius.*sind(angle_c);


head_angle = 180*rand(1,numParticles);

distribution = [zeros(1,LinearMeanSperm) ...                                  % Stright
                ones(1,numParticles-HyperSperm-DeadSperm-LinearMeanSperm)...  % Circular                 
                ones(1,HyperSperm)*2 ...                                                % Hyperactive
                ones(1,+DeadSperm)*3];                                                  % Dead                                              

Z = [];

fy = @(angle) -(sind(angle) + harmonic_R*sind(3*angle));
[~,fval] = fminbnd(fy,0,90);

Tail = [];
Tail_compile = [];
Tail_history = [];
tail_counter = zeros(1,numParticles);
tail_counter_max = ceil(FPS/10);
Tail_L = [];
Tail_hold = [];
f = waitbar(0,'Please wait...');
% if get(handles.Tail_Option,'value') == 1
%     rng(str2double(get(handles.RNG_index,'String')))
% end

%% FOR each image frame, move the particle, make the image, draw to screen,
% and save to movie file
for k = 1:numFrames
    for p = 1:numParticles
        
        if get(handles.Tail_Option,'value') == 1
            if k > 1
                Tail = Tail_hold(:,:,p);
            else
                Tail = [];
            end
        end
        
        %% LINEAR MEAN
        if distribution(p) < 1
            [x(p),y(p),x_L(1,p),y_L(1,p),head_angle(p),angle_s(p),Tail] = linear_mean_path_v2(x(p),y(p),ribbon_x(p),ribbon_y(p),...
                angle_s(p),d_angle_s(p),Direction(p),velocity(p),FPS,Rand_Displace,harmonic_R,temp,temp2,temp3,Tail_length,Tail_cycle_n);
        elseif distribution(p) == 1
        %% CIRCULAR
            [center_r_x(p),center_r_y(p),x(p),y(p),x_L(1,p),y_L(1,p),head_angle(p),angle_c(p),Tail] = circular_path_v2(x(p),y(p),radius(p),ampli(p),...
                beat_f(p),k,FPS,angle_cs(p),angle_c(p),d_angle_c(p),center_r_x(p),center_r_y(p),b,L_T,lambda,Rand_Displace);
        elseif distribution(p) == 2
        %% HYPER-ACTIVE
            [x(p),y(p),x_L(1,p),y_L(1,p),head_angle(p),Tail] = hyperactive_path_v2(x(p),y(p),Hyper_Disp,Rand_Displace,Tail,L_T,FPS);
        else
        %% DEAD
            [x(p),y(p),x_L(1,p),y_L(1,p),head_angle(p),Tail] = Dead_path_v2(x(p),y(p),head_angle(p),Hyper_Disp,Tail,L_T,FPS,Rand_Displace);
        end
        
        if get(handles.Tail_Option,'value') == 1
            if tail_counter(p) > 0
                tail_counter(p) = tail_counter(p) - 1;
                P_T = (1 - tail_counter(p)/tail_counter_max);
                T_index = ceil(P_T * length(Tail));
                Tail_L_Temp = Tail_L(:,1:end-T_index+1,p)-Tail_L(:,1,p);
                Tail = [Tail(:,1:T_index-1) Tail_L_Temp + Tail(:,T_index)];
            end
        end
        
        r(1,p) = 0;
        % Renewal of cells:
        % Only linear mean swimming sperm cells are renewed (Circular,
        % Hyperactive, or Dead cells are assumed to stay within the frame or will retun to the frame)
        if (x(p)<0 || x(p)>numPixels || y(p)<0 || y(p)>numPixels) && distribution(p) == 0
            r(1,p) = 1;
            random = rand;
            if random < 1/4
                x(p) = rand * numPixels;
                y(p) = 0;
                Direction(p) = rand*180;
                angle_s(p) = 360*rand;
            elseif random < 1/2
                x(p) = 0;
                y(p) = rand * numPixels;
                Direction(p) = rand*180 - 90;
                angle_s(p) = 360*rand;
            elseif random < 3/4
                x(p) = rand * numPixels;
                y(p) = numPixels;
                Direction(p) = -rand*180;
                angle_s(p) = 360*rand;
            else
                x(p) = numPixels;
                y(p) = rand * numPixels;
                Direction(p) = rand*180 + 90;
                angle_s(p) = 360*rand;
            end
        end

        %% Change swim type
        past_dist = distribution(p);
        
        if get(handles.SwimTransition_YN,'value') == 1 && k > 1
            
            frame_change = str2double(get(handles.Change_N_P,'string'))*FPS;
            
            LM_LM = str2double(get(handles.LM_LM,'String'));
            LM_C = str2double(get(handles.LM_C,'String'));
            LM_H = str2double(get(handles.LM_H,'String'));
            LM_D = str2double(get(handles.LM_D,'String'));

            C_LM = str2double(get(handles.C_LM,'String'));
            C_C = str2double(get(handles.C_C,'String'));
            C_H = str2double(get(handles.C_H,'String'));
            C_D = str2double(get(handles.C_D,'String'));

            H_LM = str2double(get(handles.H_LM,'String'));
            H_C = str2double(get(handles.H_C,'String'));
            H_H = str2double(get(handles.H_H,'String'));
            H_D = str2double(get(handles.H_D,'String'));

            D_LM = str2double(get(handles.D_LM,'String'));
            D_C = str2double(get(handles.D_C,'String'));
            D_H = str2double(get(handles.D_H,'String'));
            D_D = str2double(get(handles.D_D,'String'));

            Change_Prob = [LM_LM    LM_C    LM_H    LM_D;...
                            C_LM    C_C     C_H     C_D;...
                            H_LM    H_C     H_H     H_D;...
                            D_LM    D_C     D_H     D_D];
            
%                         sum(Change_Prob,'all')
            if sum(Change_Prob,'all') ~= 400
%                 close
                errordlg('Swim Type Transition Probability Distribution Set Incorrectly','Swim Transition Probability Distribution Error')
                return            
            end

            LIN_C_PROB = cumsum([LM_C    LM_H    LM_D])/100;
            CIR_C_PROB = cumsum([C_LM     C_H     C_D])/100;
            HYP_C_PROB = cumsum([H_LM     H_C     H_D])/100;
            DEA_C_PROB = cumsum([D_LM     D_C     D_H])/100; 

            
            if mod(k,frame_change) == 0
                RANDOM_TEMP = rand;
                if distribution(p) == 0             % Linear
                    if RANDOM_TEMP < LIN_C_PROB(1)
                        % Change swim type from linear to circular
                        [distribution(p),angle_c(p),radius(p),d_angle_c(p),ampli(p),beat_f(p),angle_cs(p),center_r_x(p),center_r_y(p)] = ...
                         lin2cir(Direction(p),radius_v,radius_m,var_d,ave_d,ampli_v,ampli_m,beat_v,beat_m,...
                         angle_s(p),k,FPS,x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < LIN_C_PROB(2)
                        % Change swim type from linear to hyperactive
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 2;
                    
                    elseif RANDOM_TEMP < LIN_C_PROB(3)
                        % Change swim type from linear to dead
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 3;
                    end
                 
                elseif distribution(p) == 1         % Circular
                    if RANDOM_TEMP < CIR_C_PROB(1)
                        % Change swim type from circular to linear
                        [distribution(p),velocity(p),Direction(p),ribbon_y(p),ribbon_x(p),d_angle_s(p),angle_s(p),x(p),y(p)] = ...
                         cir2lin(variance_LV,desired_LV,variance_rib_y,desired_rib_y,...
                         variance_rib_x,desired_rib_x,variance_A,desired_A,angle_c(p),d_angle_c(p),x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < CIR_C_PROB(2)
                        % Change swim type from circular to hyperactive
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 2;
                        
                    elseif RANDOM_TEMP < CIR_C_PROB(3)
                        % Change swim type from circular to dead
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 3;
                    end

                elseif distribution(p) == 2         % Hyperactive
                    if RANDOM_TEMP < HYP_C_PROB(1)
                        % Change swim type from hyperactive to linear
                        [distribution(p),velocity(p),Direction(p),ribbon_y(p),ribbon_x(p),d_angle_s(p),angle_s(p),x(p),y(p)] = ...
                        hyp_dead2lin(variance_LV,desired_LV,variance_rib_y,desired_rib_y,variance_rib_x,desired_rib_x,...
                        variance_A,desired_A,x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < HYP_C_PROB(2)
                         % Change swim type from hyperactive to circular
                        [distribution(p),angle_c(p),radius(p),d_angle_c(p),ampli(p),beat_f(p),angle_cs(p),center_r_x(p),center_r_y(p)] = ...
                        hyp_dead2cir(radius_v,radius_m,var_d,ave_d,ampli_v,ampli_m,beat_v,beat_m,k,FPS,x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < HYP_C_PROB(3)
                        % Change swim type from hyperactive to dead
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 3;
                    end                 
                    
                elseif distribution(p) == 3         % Dead
                    if RANDOM_TEMP < DEA_C_PROB(1)
                        % Change swim type from dead to linear
                        [distribution(p),velocity(p),Direction(p),ribbon_y(p),ribbon_x(p),d_angle_s(p),angle_s(p),x(p),y(p)] = ...
                        hyp_dead2lin(variance_LV,desired_LV,variance_rib_y,desired_rib_y,variance_rib_x,desired_rib_x,...
                        variance_A,desired_A,x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < DEA_C_PROB(2)
                         % Change swim type from dead to circular
                        [distribution(p),angle_c(p),radius(p),d_angle_c(p),ampli(p),beat_f(p),angle_cs(p),center_r_x(p),center_r_y(p)] = ...
                        hyp_dead2cir(radius_v,radius_m,var_d,ave_d,ampli_v,ampli_m,beat_v,beat_m,k,FPS,x_L(p),y_L(p));
                    
                    elseif RANDOM_TEMP < DEA_C_PROB(3)
                        % Change swim type from dead to hyperactive
                        x(p) = x_L(1,p);
                        y(p) = y_L(1,p);                      
                        distribution(p) = 2;
                    end                     
                    
                end
            end
        end
        if get(handles.Tail_Option,'value') == 1
            Tail_compile = [Tail_compile; Tail'];
            Tail_hold(:,:,p) = Tail(:,:);
            
            if distribution(p) ~= past_dist && distribution(p) < 2
                tail_counter(p) = tail_counter_max; 
                Tail_L(:,:,p) = Tail(:,:);
            end
        end

    end
    Z = [Z ;[1:numParticles ; x_L; y_L; head_angle; k*ones(1,numParticles); distribution; r]'];

    if get(handles.Tail_Option,'value') == 1
        Tail_history(:,:,k) = Tail_compile;
        Tail_compile = []; % Clear the compiled Tail info
    end
    waitbar(k/numFrames,f,strcat('Generating simulation data : ', num2str(round(k/numFrames*100)),'% completed'));

end
close(f)
% Option to save data as excel
if get(handles.ExcelSaveYN,'Value') == 1
   D_name = strcat(get(handles.Name_string,'String'),'.csv');
   csvwrite(D_name,Z);
%    D_name = strcat(get(handles.Name_string,'String'),'.xlsx');
%    xlswrite(D_name,Z);
end


%% Video Save option
if get(handles.save_yn,'Value') == 1
    % Open a movie file to store the movie in
    V_name = strcat(get(handles.Name_string,'String'),'.mp4');
    writerObj = VideoWriter(V_name,'MPEG-4');
    writerObj.FrameRate = ceil(str2double(get(handles.FPS,'String')));
    writerObj.Quality = 100;
    open(writerObj);
    if get(handles.Track_Option,'value') ~= 1 % Separate File to save video with tracks
        % Open a movie file to store the movie in
        V_name2 = strcat(get(handles.Name_string,'String'),'_tracks.mp4');
        writerObj2 = VideoWriter(V_name2,'MPEG-4');
        writerObj2.FrameRate = writerObj.FrameRate;
        writerObj2.Quality = 100;
        open(writerObj2);
    end
end

% Open a figure
fig = figure(999);
clf(fig);

for k = 1:numFrames
    x_L =           Z((k-1)*numParticles+1:k*numParticles,2);
    y_L =           Z((k-1)*numParticles+1:k*numParticles,3);
    head_angle =    Z((k-1)*numParticles+1:k*numParticles,4);
    
    Z_temp = Z(1:k*numParticles,:);
    
    I1 = zeros(numPixels,numPixels);
    I2 = zeros(numPixels,numPixels);
    
    I1f = zeros(numPixels,numPixels);
    I2f = zeros(numPixels,numPixels);
    
    I = I1;
    for p = 1:numParticles
        xpos = round(x_L(p));
        ypos = round(y_L(p));
        
        if (xpos>0) && (ypos>0) && (xpos<=numPixels) && (ypos<=numPixels)
            I1_temp = I1;
            I2_temp = I2;

            I1_temp(xpos,ypos) = Io1(p);
            I2_temp(xpos,ypos) = Io2(p);

            psf1r = imrotate(psf1,head_angle(p));
            psf2r = imrotate(psf2,head_angle(p));

            I1_temp(:,:) = imfilter(I1_temp(:,:), psf1r);
            I2_temp(:,:) = imfilter(I2_temp(:,:), psf2r);        

            I1f = I1f + I1_temp;
            I2f = I2f + I2_temp;
        end
    end
    
    % Add noise to the image
    I1 = I1f;
    I2 = I2f;
    I1(I1<0) = 0;
    I1(I1>1) = 1;
    
    I1 = imcomplement(I1(:,:));
    
    A(:,:) = imnoise(I1(:,:), 'gaussian', -I_High, sig_Io);
    I_diff = I_High - I_Low;
%     I_diff
    A(:,:) = A(:,:) + I_diff*repmat(linspace(0,1,numPixels),[numPixels 1]);
    A(A<0) = 0;
    A(A>1) = 1;
    
%     I_diff
%     I1(:,end)
    % Display the image
    
    I_temp = zeros(numPixels,numPixels);
    I_temp(round(numPixels/2),round(numPixels/2)) = mean(Io2);
    I_temp(:,:) = imfilter(I_temp(:,:), psf2r);
    
    B = I2(:,:);
    B(B<0) = 0;
    B = B/max(I_temp,[],'all')*Halo_Int_H*Int_H/100;
%     B = B/max(B,[],'all')*Halo_Int_H*Int_H/100;
    C = A+B;
    C(C>1) = 1;
    
%     get(handles.Tail_Option,'value')
    if get(handles.Tail_Option,'value') == 1
        Tail = Tail_history(:,:,k);
        xpos_T = round(Tail(:,1));
        ypos_T = round(Tail(:,2));
        Tail_pos = [xpos_T ypos_T];
        Tail_pos = unique(Tail_pos,'rows');
        Tail_pos = Tail_pos((Tail_pos(:,1)>0 & Tail_pos(:,1)<numPixels) & (Tail_pos(:,2)>0 & Tail_pos(:,2)<numPixels),:);

        index = sub2ind(size(I),Tail_pos(:,1),Tail_pos(:,2));
        I(index) = 1;

        I(:,:) = imfilter(I(:,:), psf3);

        I = I/max(I,[],'all')*tail_int/100;
        I = C + I;
        I(I>1) = 1;
        I(I<0) = 0;

        figure(999);
        imshow(I,'Border','tight');
    else
        figure(999);
        imshow(C,'Border','tight');
    end

    
    if get(handles.save_yn,'Value') == 1
%         pause(1)
        % Save the current frame to the movie file
        currFrame = getframe(fig);
        writeVideo(writerObj,currFrame);
    end
%     pause(0.5)
    
    hold on
    if get(handles.Track_Option,'value') == 4
        scatter(Z_temp(:,3),Z_temp(:,2),1);
    elseif get(handles.Track_Option,'value') == 3
        if k < FPS
            figure(999);
            scatter(Z_temp(:,3),Z_temp(:,2),1);
        else
            figure(999);
            scatter(Z_temp((end-numParticles*FPS+1:end),3),Z_temp((end-numParticles*FPS+1:end),2),1);
        end
    elseif get(handles.Track_Option,'value') == 2
        if k < FPS/2
            figure(999);
            scatter(Z_temp(:,3),Z_temp(:,2),1);
        else
            figure(999);
            scatter(Z_temp((end-numParticles*FPS/2+1:end),3),Z_temp((end-numParticles*FPS/2+1:end),2),1);
        end
    end
    hold off   
    % Save video with tracks option
    if get(handles.save_yn,'Value') == 1 && get(handles.Track_Option,'value') ~= 1
        % Save the current frame to the movie file
        currFrame = getframe(fig);
        writeVideo(writerObj2,currFrame);
    end
 
    set(fig,'Name',strcat(num2str(round(k/numFrames*100,2)),"% completed"),'NumberTitle','off')
%     clf

end

if get(handles.save_yn,'Value') == 1
    % Close the AVI file
    close(writerObj)
    if get(handles.Track_Option,'value') ~= 1 % Separate File to save video with tracks
        close(writerObj2)
    end
end

close(fig)
msgbox('Operation Completed')

%%
function Sample_distribution_Callback(hObject, eventdata, handles)


V = str2double(get(handles.Vertical_Size,'String'));
H = str2double(get(handles.Horizontal_Size,'String'));
Halo_Ratio = str2double(get(handles.Halo_Ratio,'String'));

Int_H = str2double(get(handles.Intensity_percent_H,'String'));
Halo_Int_H = str2double(get(handles.Halo_Intensity,'String'))/100;

% Image noise 
sig_Io = str2double(get(handles.Noise_Level,'String'));

I_High = 1 - str2double(get(handles.Background_I,'String'))/100;
I_Low = 1 - str2double(get(handles.Background_I_L,'String'))/100;



%% Circular motion sperm
%particle beating frequency/amplitude
beat_v = str2double(get(handles.Beat_v,'String'));
beat_m = str2double(get(handles.Beat_m,'String'));
ampli_v = str2double(get(handles.Amp_v,'String'));
ampli_m = str2double(get(handles.Amp_m,'String'));

beat_f = abs(beat_v*(randn)+beat_m);
ampli = ampli_v*randn+ampli_m;

%radius of each particle spin
radius_m = str2double(get(handles.Radius_m,'String'));
radius_v = str2double(get(handles.Radius_v,'String'));
radius = radius_v*(randn)+radius_m;

%average change of angle and its variance
ave_d = str2double(get(handles.AV_m,'String'));
var_d = str2double(get(handles.AV_v,'String'));

%starting angles for each particles
angle_c = 90;

d_angle_c = -(var_d*abs(randn(1,1))+ave_d);

angle_cs = 80;

%% Tail parameters
Tail_L = str2double(get(handles.Tail_L,'String'));
tail_int  = str2double(get(handles.Tail_Int,'String'));
Tail_width = str2double(get(handles.Tail_W_D,'String'));

% CIRCULAR TAIL PARAMETERS
b = @(x) 0.02*x + 0.8;
lambda = Tail_L;

L_T = linspace(0,lambda,300);

%% Set filter (point spread function for head shape)
mu = [0 0];
sigma = [H 0;0 V];

if and(H<5,V<5)
    x1 = -12:12;
    x2 = -12:12;
elseif and(H<10,V<10)
    x1 = -25:25;
    x2 = -25:25;
else
    x1 = -100:100;
    x2 = -100:100;
end

[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

psf1 = mvnpdf(X,mu,sigma);
psf1 = reshape(psf1,length(x2),length(x1));
psf1 = psf1/sum(psf1,'all');

psf2 = mvnpdf(X,mu,sigma*Halo_Ratio);
psf2 = reshape(psf2,length(x2),length(x1));
psf2 = psf2/sum(psf2,'all');
psf2 = del2(psf2);

Intensity = 1/max(psf1,[],'all')*Int_H/100;

psf3 = fspecial('log', 101, Tail_width);
% figure; subplot(1,2,1); surf(psf3);
% psf3 = -psf3;
% subplot(1,2,2); surf(psf3)
% psf3 = psf3/sum(psf3,'all');
% figure; surf(psf3)

%%
I1 = zeros(80,80);
I2 = zeros(80,80);
I = I1;

I1(50,40) = Intensity;
I2(50,40) = Intensity;  

I1(:,:) = imfilter(I1(:,:), psf1);
I2(:,:) = imfilter(I2(:,:), psf2);

I_no_n = I1;

I1 = imcomplement(I1(:,:));

A(:,:) = imnoise(I1(:,:), 'gaussian', -I_High, sig_Io);
I_diff = I_High - I_Low;

A(:,:) = A(:,:) - I_diff*repmat(linspace(0,1,80),[80 1]);
A(A<0) = 0;
A(A>1) = 1;
% temp = psf2;
% temp(temp<0) = 0;
% size(X1)
% figure; surf(X1,X2,psf3)

% % Display the image
% A = imcomplement(I1(:,:));
B = I2(:,:);

B = B/max(B,[],'all')*Halo_Int_H*Int_H/100;
B(B<0) = 0;
C = A+B;
C(C>1) = 1;

Tail = C_Tail(50,40,ampli(1),beat_f(1),1,120,angle_cs(1),angle_c(1),d_angle_c(1),b,L_T,lambda);

xpos_T = round(Tail(1,:));
ypos_T = round(Tail(2,:));
Tail_pos = [xpos_T' ypos_T'];
Tail_pos = unique(Tail_pos,'rows');
Tail_pos = Tail_pos((Tail_pos(:,1)>0 & Tail_pos(:,1)<120) & (Tail_pos(:,2)>0 & Tail_pos(:,2)<120),:);

index = sub2ind(size(I),Tail_pos(:,1),Tail_pos(:,2));
I(index) = 1;
I(:,:) = imfilter(I(:,:), psf3);

I = I/max(I,[],'all')*tail_int/100;

I2 = C + I;
I2(I2>1) = 1;
I2(I2<0) = 0;

figure; 
subplot(2,4,1); imshow(I_no_n);     title('I_3')
subplot(2,4,2); imshow(I1);         title('I_4')
subplot(2,4,3); imshow(A);          title('I_5')
subplot(2,4,4); imshow(B);          title('I_6')
subplot(2,4,5); imshow(C);          title('I_7')
subplot(2,4,6); imshow(I);          title('I_8')
subplot(2,4,7); imshow(I2);         title('I_9')
set(gcf,'Position',[100 500 950 300])

figure; 
subplot(2,4,1); surf(uint8(I_no_n*255),'edgecolor','none');     title('I_3'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse'), caxis([0 255])
subplot(2,4,2); surf(uint8(I1*255),'edgecolor','none');         title('I_4'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse'), caxis([0 255])
subplot(2,4,3); surf(uint8(A*255),'edgecolor','none');          title('I_5'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse'), caxis([0 255])
subplot(2,4,4); surf(uint8(B*255),'edgecolor','none');          title('I_6'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse')
subplot(2,4,5); surf(uint8(C*255),'edgecolor','none');          title('I_7'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse'), caxis([0 255])
subplot(2,4,6); surf(I*255,'edgecolor','none');          title('I_8'); zlim([-0.2*255 255]), view(2), colorbar, set(gca,'ydir','reverse')
subplot(2,4,7); surf(uint8(I2*255),'edgecolor','none');         title('I_9'); zlim([0 255]), view(2), colorbar, set(gca,'ydir','reverse'), caxis([0 255])
set(gcf,'Position',[100 100 950 300])

% figure;
% imshow(I2); hold on; scatter(ypos_T,xpos_T,7,'o','filled'); scatter(40,50,15,'g','filled');

% --- Executes on button press in SwimTransition_YN.
function SwimTransition_YN_Callback(hObject, eventdata, handles)
% hObject    handle to SwimTransition_YN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SwimTransition_YN
if get(handles.SwimTransition_YN,'value') == 1
    set(handles.Change_N_P,'Enable','on')
    set(handles.LM_LM,'Enable','on')
    set(handles.LM_C,'Enable','on')
    set(handles.LM_H,'Enable','on')
    set(handles.LM_D,'Enable','on')

    set(handles.C_LM,'Enable','on')
    set(handles.C_C,'Enable','on')
    set(handles.C_H,'Enable','on')
    set(handles.C_D,'Enable','on')

    set(handles.H_LM,'Enable','on')
    set(handles.H_C,'Enable','on')
    set(handles.H_H,'Enable','on')
    set(handles.H_D,'Enable','on')

    set(handles.D_LM,'Enable','on')
    set(handles.D_C,'Enable','on')
    set(handles.D_H,'Enable','on')
    set(handles.D_D,'Enable','on')
    
else
    set(handles.Change_N_P,'Enable','off')    
    set(handles.LM_LM,'Enable','off')
    set(handles.LM_C,'Enable','off')
    set(handles.LM_H,'Enable','off')
    set(handles.LM_D,'Enable','off')

    set(handles.C_LM,'Enable','off')
    set(handles.C_C,'Enable','off')
    set(handles.C_H,'Enable','off')
    set(handles.C_D,'Enable','off')

    set(handles.H_LM,'Enable','off')
    set(handles.H_C,'Enable','off')
    set(handles.H_H,'Enable','off')
    set(handles.H_D,'Enable','off')

    set(handles.D_LM,'Enable','off')
    set(handles.D_C,'Enable','off')
    set(handles.D_H,'Enable','off')
    set(handles.D_D,'Enable','off')
end


function HARMONIC_SLIDER_Callback(hObject, eventdata, handles)

temp = get(handles.HARMONIC_SLIDER,'value');
set(handles.harmonic_R,'String',round(temp,2));
set(handles.HARMONIC_SLIDER,'String',round(temp,2));

function Halo_Intensity_S_Callback(hObject, eventdata, handles)

temp = get(handles.Halo_Intensity_S,'value')*100;
set(handles.Halo_Intensity,'String',round(temp,3));
set(handles.Halo_Intensity_S,'value',round(temp,3)/100);

function IntensityRange_H_Callback(hObject, eventdata, handles)

temp_H = get(handles.IntensityRange_H,'value');
temp_L = get(handles.IntensityRange_L,'value');
set(handles.Intensity_percent_L,'String',round(temp_H*temp_L*99+1));
set(handles.Intensity_percent_H,'String',round(temp_H*99+1));
set(handles.IntensityRange_H,'String',round(temp_H,2));

function IntensityRange_L_Callback(hObject, eventdata, handles)

temp_H = get(handles.IntensityRange_H,'value');
temp_L = get(handles.IntensityRange_L,'value');
set(handles.Intensity_percent_L,'String',round(temp_H*temp_L*99+1));
set(handles.IntensityRange_L,'String',round(temp_L,2));

function Horizontal_Size_S_Callback(hObject, eventdata, handles)

temp = get(handles.Horizontal_Size_S,'value');
set(handles.Horizontal_Size,'String',round(temp*49+1));
set(handles.Horizontal_Size_S,'String',round(temp,2));

function Vertical_Size_S_Callback(hObject, eventdata, handles)

temp = get(handles.Vertical_Size_S,'value');
set(handles.Vertical_Size,'String',round(temp*49+1));
set(handles.Vertical_Size_S,'String',round(temp,2));

function Halo_Ratio_S_Callback(hObject, eventdata, handles)

temp = get(handles.Halo_Ratio_S,'value');
set(handles.Halo_Ratio,'String',round(temp*9+1,1));
set(handles.Halo_Ratio_S,'String',round(temp,3));

function Tail_L_S_Callback(hObject, eventdata, handles)

temp = get(handles.Tail_L_S,'value');
set(handles.Tail_L,'String',round(temp*99+1,0));
set(handles.Tail_L_S,'String',round(temp,3));

function Tail_Int_S_Callback(hObject, eventdata, handles)

temp = get(handles.Tail_Int_S,'value');
set(handles.Tail_Int,'String',round(temp*99+1,0));
set(handles.Tail_Int_S,'String',round(temp,3));

function Background_I_S_Callback(hObject, eventdata, handles)
temp_H = get(handles.Background_I_S,'value');
temp_L = get(handles.Background_I_L_S,'value');

set(handles.Background_I,'String',round(temp_H*99+1,0));
set(handles.Background_I_S,'String',round(temp_H,3));
set(handles.Background_I_L,'String',round(temp_H*temp_L*99+1,0));

% --- Executes on slider movement.
function Background_I_L_S_Callback(hObject, eventdata, handles)
temp_H = get(handles.Background_I_S,'value');
temp_L = get(handles.Background_I_L_S,'value');

set(handles.Background_I_L,'String',round(temp_H*temp_L*99+1,0));
set(handles.Background_I_L_S,'String',round(temp_L,2));

function Tail_W_D_S_Callback(hObject, eventdata, handles)
temp = get(handles.Tail_W_D_S,'value');
set(handles.Tail_W_D,'String',round(temp*10,1));
set(handles.Tail_W_D_S,'String',round(temp,3));




function RNG_setting_Callback(hObject, eventdata, handles)
if get(handles.RNG_setting,'value') == 1
    set(handles.RNG_index,'Enable','on')
else
    set(handles.RNG_index,'Enable','off')
end

%%
function particle_n_Callback(hObject, eventdata, handles)

function particle_n_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Frame_size_Callback(hObject, eventdata, handles)

function Frame_size_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Dead_P_Callback(hObject, eventdata, handles)

function Dead_P_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Hyper_P_Callback(hObject, eventdata, handles)

function Hyper_P_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Noise_YN_Callback(hObject, eventdata, handles)

function popupmenu1_Callback(hObject, eventdata, handles)

function popupmenu1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Video_time_Callback(hObject, eventdata, handles)

function Video_time_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Noise_Level_Callback(hObject, eventdata, handles)

function Noise_Level_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Straight_Dis_Callback(hObject, eventdata, handles)

function Straight_Dis_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function save_yn_Callback(hObject, eventdata, handles)


function Name_string_Callback(hObject, eventdata, handles)

function Name_string_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Beat_m_Callback(hObject, eventdata, handles)

function Beat_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Beat_v_Callback(hObject, eventdata, handles)

function Beat_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Amp_m_Callback(hObject, eventdata, handles)

function Amp_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Amp_v_Callback(hObject, eventdata, handles)

function Amp_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Radius_m_Callback(hObject, eventdata, handles)

function Radius_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Radius_v_Callback(hObject, eventdata, handles)

function Radius_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function AV_m_Callback(hObject, eventdata, handles)

function AV_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function AV_v_Callback(hObject, eventdata, handles)

function AV_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LV_m_Callback(hObject, eventdata, handles)

function LV_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LV_v_Callback(hObject, eventdata, handles)

function LV_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function width_m_Callback(hObject, eventdata, handles)

function width_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function width_v_Callback(hObject, eventdata, handles)

function width_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function height_m_Callback(hObject, eventdata, handles)

function height_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function height_v_Callback(hObject, eventdata, handles)

function height_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LV_a_m_Callback(hObject, eventdata, handles)

function LV_a_m_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LV_a_v_Callback(hObject, eventdata, handles)

function LV_a_v_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function FPS_Callback(hObject, eventdata, handles)

function FPS_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ExcelSaveYN_Callback(hObject, eventdata, handles)

function IntensityRange_H_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function Intensity_percent_H_Callback(hObject, eventdata, handles)

function Intensity_percent_H_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function IntensityRange_L_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function Intensity_percent_L_Callback(hObject, eventdata, handles)

function Intensity_percent_L_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Horizontal_Size_S_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function Horizontal_Size_Callback(hObject, eventdata, handles)

function Horizontal_Size_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Vertical_Size_S_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function Vertical_Size_Callback(hObject, eventdata, handles)

function Vertical_Size_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Halo_Ratio_S_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Halo_Ratio_Callback(hObject, eventdata, handles)

function Halo_Ratio_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Halo_Intensity_S_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function Halo_Intensity_Callback(hObject, eventdata, handles)

function Halo_Intensity_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Track_Option_Callback(hObject, eventdata, handles)

function Track_Option_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Tail_Option_Callback(hObject, eventdata, handles)


function Hyper_Disp_Callback(hObject, eventdata, handles)

function Hyper_Disp_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Rand_Displace_Callback(hObject, eventdata, handles)

function Rand_Displace_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LM_LM_Callback(hObject, eventdata, handles)

function LM_LM_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C_LM_Callback(hObject, eventdata, handles)

function C_LM_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function H_LM_Callback(hObject, eventdata, handles)

function H_LM_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function D_LM_Callback(hObject, eventdata, handles)

function D_LM_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LM_C_Callback(hObject, eventdata, handles)

function LM_C_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C_C_Callback(hObject, eventdata, handles)

function C_C_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function H_C_Callback(hObject, eventdata, handles)

function H_C_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function D_C_Callback(hObject, eventdata, handles)

function D_C_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LM_H_Callback(hObject, eventdata, handles)

function LM_H_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C_H_Callback(hObject, eventdata, handles)

function C_H_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function H_H_Callback(hObject, eventdata, handles)

function H_H_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function D_H_Callback(hObject, eventdata, handles)

function D_H_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function LM_D_Callback(hObject, eventdata, handles)

function LM_D_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function C_D_Callback(hObject, eventdata, handles)

function C_D_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function H_D_Callback(hObject, eventdata, handles)

function H_D_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function D_D_Callback(hObject, eventdata, handles)

function D_D_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Change_N_P_Callback(hObject, eventdata, handles)

function Change_N_P_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function SwimTypeChange_N_Callback(hObject, eventdata, handles)

function SwimTypeChange_N_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function RNG_index_Callback(hObject, eventdata, handles)

function RNG_index_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function harmonic_R_Callback(hObject, eventdata, handles)

function harmonic_R_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function HARMONIC_SLIDER_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes during object creation, after setting all properties.
function Tail_L_S_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_L_S (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes during object creation, after setting all properties.
function Tail_Int_S_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_Int_S (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes during object creation, after setting all properties.
function Background_I_S_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Background_I_S (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Tail_L_Callback(hObject, eventdata, handles)
% hObject    handle to Tail_L (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Tail_L as text
%        str2double(get(hObject,'String')) returns contents of Tail_L as a double


% --- Executes during object creation, after setting all properties.
function Tail_L_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_L (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Tail_Int_Callback(hObject, eventdata, handles)
% hObject    handle to Tail_Int (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Tail_Int as text
%        str2double(get(hObject,'String')) returns contents of Tail_Int as a double


% --- Executes during object creation, after setting all properties.
function Tail_Int_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_Int (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Background_I_Callback(hObject, eventdata, handles)
% hObject    handle to Background_I (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Background_I as text
%        str2double(get(hObject,'String')) returns contents of Background_I as a double


% --- Executes during object creation, after setting all properties.
function Background_I_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Background_I (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function Tail_W_D_S_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_W_D_S (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Tail_W_D_Callback(hObject, eventdata, handles)
% hObject    handle to Tail_W_D (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Tail_W_D as text
%        str2double(get(hObject,'String')) returns contents of Tail_W_D as a double


% --- Executes during object creation, after setting all properties.
function Tail_W_D_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tail_W_D (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function Background_I_L_S_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Background_I_L_S (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Background_I_L_Callback(hObject, eventdata, handles)
% hObject    handle to Background_I_L (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Background_I_L as text
%        str2double(get(hObject,'String')) returns contents of Background_I_L as a double


% --- Executes during object creation, after setting all properties.
function Background_I_L_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Background_I_L (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
