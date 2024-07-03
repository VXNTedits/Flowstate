theta = -89:10:90;
phi = -90:10:90;
[theta, phi] = meshgrid(theta, phi);

l = sqrt( (cosd(theta).*cosd(phi)).^2.*(sind(theta)).^2.*(cosd(theta).*sind(phi)).^2 )

surf(theta, phi, l);