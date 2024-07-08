
pitch = [-0.14, 1.31, 1.22, 1.5, -0.99, 0.77, 1.13, -0.84, 1.26, -0.9, 1.34]
yaw = [1.88, -4.48, 2.09, 2.47, 2.55, 2.56, 3.69, 2.09, 2.49, 2.23, 2.91]

theta = [-0.32, -1.62, -1.56, -1.79, 0.87, -1.2, -1.48, 0.58, -1.59, 0.66, -1.65]
phi = [0.08, 6.8, 0.23, 0.19, 0.12, -0.47, -1.42, 0.18, -0.13, 0.13, -1.47]

phi_corr = phi + yaw

a = 0.5
t = pi()*0.6
c = pi()*1.4
d = 2.3

%plot(pitch, theta,"o")
ylabel('theta')
hold on 
plot(pitch, phi_corr,"o")
hold on
x = linspace(min(pitch), max(pitch), 100);
y = a*sin(t*x+c)+d;
plot(x, y, '--', 'DisplayName', 'Cosine Function');
