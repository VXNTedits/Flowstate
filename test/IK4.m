pitch = -90:2:90;
yaw   = -180:4:180;

vx = cosd(pitch).*cosd(yaw);
vy = sind(pitch);
vz = cosd(pitch).*sind(yaw);

l   =  0.812;
d_y = -0.314;
d_z = -0.25;

theta = -90:2:90;
phi   = -180:4:180;

ax = cosd(theta).*cosd(phi);
ay = sind(theta);
az = cosd(theta).*sind(phi);

zeta = (ax.*vx+ay.*vy+az.*vz)/(vx.^2+vy.^2+vz.^2);
e = sqrt((zeta.*vx-ax).^2+(zeta.*vy-ay).^2+(zeta.*vz-az).^2);

plot3(pitch, yaw, e)