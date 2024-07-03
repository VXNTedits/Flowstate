view_pitch = -89:5:89;
view_yaw = -90:5:90;
[view_pitch, view_yaw] = meshgrid(view_pitch, view_yaw);

x0 = cosd(view_pitch) .* cosd(view_yaw);
y0 = sind(view_pitch);
z0 = cosd(view_pitch) .* sind(view_yaw);

eye_to_shoulder = 0.314;
arm_length = 0.812;
shoulder_offset = 0.25;

dx = shoulder_offset .* cosd(view_yaw);
dy = -eye_to_shoulder;
dz = shoulder_offset .* sind(view_yaw);
l = arm_length;

A = dy - dx .* y0;
B = dy - dz .* y0;
E = l .* y0 .^2;
C = (x0 .* l) .^2;
Ap = 2 .* x0 .* l .* A;
D = (z0 .* l) .^2;
Bp = 2 .* z0 .* l .* B;
alpha = -E .^2 - C - D;
beta = Ap - Bp;
gamma = E .^2 - A .^2 - B .^2;

pitch_angle = asin((-beta + sqrt(beta .^2 - 4 .* alpha .* gamma + 0j)) ./ (2 .* alpha));

% To avoid potential dimension mismatches, ensure that all operations are element-wise
term1 = (l .* sin(pitch_angle) + dz) ./ (l .* z0 .* cos(pitch_angle));
term2 = dy ./ (l .* cos(pitch_angle));
yaw_angle = asin(y0 .* term1 - term2);

% Plotting both angles using subplot
figure;

% Plot pitch_angle
subplot(1, 2, 1); % 1 row, 2 columns, first subplot
surf(view_pitch, view_yaw, real(pitch_angle));
title('Pitch Angle');
xlabel('View Pitch');
ylabel('View Yaw');
zlabel('Pitch Angle');
%shading interp;

% Plot yaw_angle
subplot(1, 2, 2); % 1 row, 2 columns, second subplot
surf(view_pitch, view_yaw, real(yaw_angle));
title('Yaw Angle');
xlabel('View Pitch');
ylabel('View Yaw');
zlabel('Yaw Angle');
%shading interp;
