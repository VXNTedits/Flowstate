
pitch = linspace(-90,90,3);
yaw   = linspace(-180,180,3);

l   =  0.812;
d_y = -0.314;
d_z = -0.25;

the = linspace(-90,90,100);
phi   = linspace(-180,180,100);
[the, phi] = meshgrid(the, phi);

%dedx=(l^2 * ((v_z^2 + v_y^2) * a_x - a_z * v_x * v_z - a_y * v_x * v_y)) / ((v_z^2 + v_y^2 + v_x^2) * sqrt(((v_z * (l * v_x * a_x + a_z * l * v_z + a_y * l * v_y)) / (v_z^2 + v_y^2 + v_x^2) - a_z * l)^2 + ((v_y * (l * v_x * a_x + a_z * l * v_z + a_y * l * v_y)) / (v_z^2 + v_y^2 + v_x^2) - a_y * l)^2 + ((v_x * (l * v_x * a_x + a_z * l * v_z + a_y * l * v_y)) / (v_z^2 + v_y^2 + v_x^2) - l * a_x)^2));
%dedy=(l^2 * ((v_z^2 + v_x^2) * a_y - a_z * v_y * v_z - a_x * v_x * v_y)) / ((v_z^2 + v_y^2 + v_x^2) * sqrt(((v_z * (l * v_y * a_y + a_z * l * v_z + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - a_z * l)^2 + ((v_y * (l * v_y * a_y + a_z * l * v_z + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - l * a_y)^2 + ((v_x * (l * v_y * a_y + a_z * l * v_z + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - a_x * l)^2));
%dedz=(l^2 * ((v_y^2 + v_x^2) * a_z + (-a_y * v_y - a_x * v_x) * v_z)) / ((v_z^2 + v_y^2 + v_x^2) * sqrt(((v_z * (l * v_z * a_z + a_y * l * v_y + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - l * a_z)^2 + ((v_y * (l * v_z * a_z + a_y * l * v_y + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - a_y * l)^2 + ((v_x * (l * v_z * a_z + a_y * l * v_y + a_x * l * v_x)) / (v_z^2 + v_y^2 + v_x^2) - a_x * l)^2));

figure;

for i = 1:length(pitch)
    for j = 1:length(yaw)

        v_x = cosd(pitch).*cosd(yaw);
        v_y = sind(pitch);
        v_z = cosd(pitch).*sind(yaw);

        dedthe=(l^2 * ((v_y * cosd(v_y * the) + (-sind(v_z * phi) - cosd(v_x * phi) + v_z * sind(phi) + v_x * cosd(phi)) * sind(the) - v_y * cosd(the)) * sind(v_y * the) + ((v_y * sind(v_z * phi) + v_y * cosd(v_x * phi) - v_y * v_z * sind(phi) - v_x * v_y * cosd(phi)) * cosd(the) - v_y^2 * sind(the)) * cosd(v_y * the) + (v_y * sind(v_z * phi) + v_y * cosd(v_x * phi)) * sind(the)^2 + (-sind(v_z * phi)^2 + (-2 * cosd(v_x * phi) + 2 * v_z * sind(phi) + 2 * v_x * cosd(phi)) * sind(v_z * phi) - cosd(v_x * phi)^2 + (2 * v_z * sind(phi) + 2 * v_x * cosd(phi)) * cosd(v_x * phi) + (-v_z^2 - v_y^2 - v_x^2) * sind(phi)^2 + (-v_z^2 - v_y^2 - v_x^2) * cosd(phi)^2 + v_z^2 + v_y^2 + v_x^2) * cosd(the) * sind(the) + (-v_y * sind(v_z * phi) - v_y * cosd(v_x * phi)) * cosd(the)^2)) / ((v_z^2 + v_y^2 + v_x^2) * sqrt(((v_z * (l * sind(v_y * the) + l * sind(v_z * phi) * cosd(the) + l * cosd(v_x * phi) * cosd(the))) / (v_z^2 + v_y^2 + v_x^2) - l * sind(phi) * cosd(the))^2 + ((v_y * (l * sind(v_y * the) + l * sind(v_z * phi) * cosd(the) + l * cosd(v_x * phi) * cosd(the))) / (v_z^2 + v_y^2 + v_x^2) - l * sind(the))^2 + ((v_x * (l * sind(v_y * the) + l * sind(v_z * phi) * cosd(the) + l * cosd(v_x * phi) * cosd(the))) / (v_z^2 + v_y^2 + v_x^2) - l * cosd(phi) * cosd(the))^2));
        dedphi=(l^2 * cosd(the) * ((v_z * cosd(the) * cosd(v_z * phi) - v_x * cosd(the) * sind(v_x * phi) + v_x * cosd(the) * sind(phi) - v_z * cosd(the) * cosd(phi)) * sind(v_z * phi) + (v_z * cosd(the) * cosd(v_x * phi) - v_z^2 * cosd(the) * sind(phi) - v_x * v_z * cosd(the) * cosd(phi) + v_z * sind(v_y * the) - v_y * v_z * sind(the)) * cosd(v_z * phi) + (-v_x * cosd(the) * cosd(v_x * phi) + v_x * v_z * cosd(the) * sind(phi) + v_x^2 * cosd(the) * cosd(phi) - v_x * sind(v_y * the) + v_x * v_y * sind(the)) * sind(v_x * phi) + (v_x * cosd(the) * sind(phi) - v_z * cosd(the) * cosd(phi)) * cosd(v_x * phi) + v_x * sind(v_y * the) * sind(phi) - v_z * sind(v_y * the) * cosd(phi))) / ((v_z^2 + v_y^2 + v_x^2) * sqrt(((v_z * (l * cosd(the) * sind(v_z * phi) + l * cosd(the) * cosd(v_x * phi) + l * sind(v_y * the))) / (v_z^2 + v_y^2 + v_x^2) - l * cosd(the) * sind(phi))^2 + ((v_y * (l * cosd(the) * sind(v_z * phi) + l * cosd(the) * cosd(v_x * phi) + l * sind(v_y * the))) / (v_z^2 + v_y^2 + v_x^2) - l * sind(the))^2 + ((v_x * (l * cosd(the) * sind(v_z * phi) + l * cosd(the) * cosd(v_x * phi) + l * sind(v_y * the))) / (v_z^2 + v_y^2 + v_x^2) - l * cosd(the) * cosd(phi))^2));
        
        subplot(length(pitch),length(yaw),(i-1)*length(yaw)+j)
        surf(the,phi,dedthe)
        

    end
end