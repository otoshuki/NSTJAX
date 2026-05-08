%% Drone exogeneous system tracking
clear
clc
addpath("NST_MATLAB/")
addpath("NST_MATLAB/npy-matlab/")

%% Overall parameters
d = 2; %Degree of solution
g = 9.81; %Gravity
alp_stable = 0.4; %Stable embedding
e3 = [0;0;1];

%% System
%sys refers to system parameters/variables
%x, u, fsym is the final system
n = 10; %System
m = 4; %Control
%State variables: 
%They are separated to make it easy to implement
xsys = sym('xsys', [3,1]); %Position
vsys = sym('vsys', [3,1]); %Velocity
psys = sym('psys', [3,1]); %P transformation
tsys = sym('tsys', [1,1]); %Yaw/2, theta

%Control variables
u0sys = sym('u0sys', [1,1]); %Thrust log
omsys = sym('omsys', [3,1]); %Spatial angular velocity

%System dynamics
xsysdot = vsys;
vsysdot = -g*e3 + (u0sys*g+g)*psys;
psysdot = cross(omsys, psys) - alp_stable*(psys(1)^2 + psys(2)^2 + psys(3)^2 - 1)*psys;
tsysdot = sum((psys+e3).*omsys)/(2*(1+psys(3)));

%Combined system
x = [xsys; vsys; psys; tsys]; %System state
u = [u0sys;omsys]; %System control
fsym = [xsysdot; vsysdot; psysdot; tsysdot]; %System dynamics

%% Exosystem Different
%traj refers to exosystem variables
%x_, f_sym is the final exosystem
n_ = 4*6;
%Exosystem parameters
traj = sym('traj', [4,1]);
traj_1dot = sym('traj_dot', [4,1]);
traj_2dot = sym('traj_2dot', [4,1]);
traj_3dot = sym('traj_3dot', [4,1]);
traj_4dot = sym('traj_4dot', [4,1]);
traj_5dot = sym('traj_5dot', [4,1]);

%Combined exosystem
x_ = [traj; traj_1dot; traj_2dot; traj_3dot; traj_4dot; traj_5dot];
f_sym = [traj_1dot; traj_2dot; traj_3dot; traj_4dot; traj_5dot; zeros(4,1)];

%% Output
p = 4;
hsym = [traj(1:3)-x(1:3); traj(4)-2*x(10)]; %theta = x(10) = yaw/2

%% Lagrangian
Q = 1e-2*diag([10, 10, 10, 0, 0, 0, 0, 0, 0, 10]);
R = 1e-2*diag([10, 10, 10, 10]);
lsym = (x.'*Q*x + u.'*R*u)/2; 

%% System operating point
x0 = [zeros(3,1); zeros(3,1); 0; 0; 1; 0]; %System operating point
u0 = zeros(m,1); %Control operating point
x_0 = zeros(n_,1);

%% Scale
xscale = ones(n,1);
uscale = ones(m,1);
x_scale = ones(n_,1);

%% Set up HJB
tic
%Taylor expansion for each part (returns coefficients in order)
[f,l,f_]=hjb_set_up(fsym,lsym,x,u,x0,u0,....
    xscale,uscale,n,m,d,f_sym,n_,x_,x_0,x_scale);
setuptime = toc

%% Set up output
%1. Scale output, not required here
hsym=subs(hsym,[x;u;x_],[x0+x.*xscale;u0+u.*uscale;x_0+x_.*x_scale]);
%2. Taylor expansion for h (returns coefficients in order)
h=tay_poly(hsym,x,u,x_,x0,u0,x_0,p,n,m,n_,d);
h=prt(h,[p,n;0,m;0,n_],[0,d],[1,d]);

%% Solve FBI equations
tic
[th,la,ff,hh]=fbi(f,h,f_,n,m,p,n_,d);
fbitime = toc

%% Batch test vectors for the JAX comparison
rng(0)
S = 64;
Xtest = randn(n_, S);
Theta_vals = zeros(n, S);
Lambda_vals = zeros(m, S);
for j = 1:S
    mj = mon(Xtest(:,j), n_, [1,d]);
    Theta_vals(:,j) = th*mj;
    Lambda_vals(:,j) = la*mj;
end
writeNPY(Xtest', "x_test_batch.npy");
writeNPY(Theta_vals', "th_val_mat.npy");
writeNPY(Lambda_vals', "la_val_mat.npy");