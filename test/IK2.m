xv = -1:0.1:1;
yv = -1:0.1:1;
zv = -1:0.1:1;

pitch = asin(yv);
yaw = asin(zv./cos(pitch));

dx = 0.25.*cos(yaw);
dy = -0.314;
dz = 0.25.*sin(yaw);

l = 0.812;
t = 0:0.5:10;

x = (- dx + t.*xv)./l;
y = (- dy + t.*yv)./l;
z = (- dz + t.*zv)./l;

e = sqrt( (t.*xv-(l.*x-dx)).^2+(t.*yv-(l.*y-dy)).^2+(t.*zv-(l.*z-dz)).^2 );

theta = asin(y);
phi = asin(z./cos(theta));

% Separate real and imaginary parts of e
e_real = real(e);
e_imag = imag(e);

% Create a 3D plot
figure;
hold on;
plot3(t, e_real, e_imag, 'b');
xlabel('t');
ylabel('Real part of e');
zlabel('Imaginary part of e');
title('3D Plot of Real and Imaginary Parts of e');
grid on;
figure;
plot3(x,y,z)
xlabel("x")
ylabel("y")
zlabel("z")
title('a span over t domain');
hold off;
