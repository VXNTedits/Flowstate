import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updating the dataset to include all the provided values
data_updated = {
    'pitch': [-0.14, 1.31, 1.22, 1.5, -0.99, 0.77, 1.13, -0.84, 1.26, -0.9, 1.34, 1.21, -0.02, -0.79, -0.7, -0.14, 0.78, 0.57, 0.37, 0.18, 0.01, -0.15, -0.3, -0.44],
    'yaw': [1.88, -4.48, 2.09, 2.47, 2.55, 2.56, 3.69, 2.09, 2.49, 2.23, 2.91, 2.58, 2.5, 2.25, 2.61, 2.07, 2.79, 3.14, 2.44, 2.34, 2.39, 2.13, 2.37, 2.39],
    'theta': [-0.34, -1.62, -1.56, -1.79, 0.87, -1.2, -1.48, 0.58, -1.59, 0.66, -1.65, -1.55, -0.49, 0.49, 0.35, -0.35, -1.22, -1.05, -0.87, -0.7, -0.52, -0.35, -0.17, 0.0],
    'phi': [0.08, 6.8, 0.23, 0.19, 0.12, -0.47, -1.42, 0.18, -0.13, 0.13, -0.46, -0.26, -0.54, -0.05, -0.49, -0.1, -0.7, -1.09, -0.44, -0.36, -0.42, -0.15, -0.39, -0.39]
}

df_updated = pd.DataFrame(data_updated)

# Sorting by 'pitch'
df_sorted_updated = df_updated.sort_values(by='pitch').reset_index(drop=True)

# Creating the 'phi_corr' field
df_sorted_updated['phi_corr'] = df_sorted_updated['phi'] + df_sorted_updated['yaw']

# Extracting the relevant columns for polynomial fit
pitch = df_sorted_updated['pitch']
theta = df_sorted_updated['theta']

# Fitting a polynomial function to the data
polynomial_coefficients_theta = np.polyfit(pitch, theta, deg=3)  # You can try different degrees

# Creating the polynomial function
polynomial_function_theta = np.poly1d(polynomial_coefficients_theta)

# Generating data points for the polynomial fit
pitch_fit_poly = np.linspace(min(pitch), max(pitch), 500)
theta_poly_fit = polynomial_function_theta(pitch_fit_poly)

# Plotting the polynomial fit results for theta as a function of pitch
plt.figure(figsize=(10, 6))
plt.scatter(pitch, theta, label='Data Points', color='blue')
plt.plot(pitch_fit_poly, theta_poly_fit, label='Polynomial Fit', color='red')
plt.xlabel('Pitch')
plt.ylabel('Theta')
plt.title('Polynomial Fit for Theta as a Function of Pitch')
plt.legend()
plt.grid(True)
plt.show()

# Display polynomial coefficients for theta as a function of pitch
print(f"Polynomial Coefficients for Theta as a Function of Pitch: {polynomial_coefficients_theta}")

# Now the previous polynomial fit for phi_corr as a function of theta

# Extracting the relevant columns for polynomial fit
theta = df_sorted_updated['theta']
phi_corr = df_sorted_updated['phi_corr']

# Fitting a polynomial function to the data
polynomial_coefficients_phi_corr = np.polyfit(theta, phi_corr, deg=4)  # You can try different degrees

# Creating the polynomial function
polynomial_function_phi_corr = np.poly1d(polynomial_coefficients_phi_corr)

# Generating data points for the polynomial fit
theta_fit_poly = np.linspace(min(theta), max(theta), 500)
phi_corr_poly_fit = polynomial_function_phi_corr(theta_fit_poly)

# Plotting the polynomial fit results for phi_corr as a function of theta
plt.figure(figsize=(10, 6))
plt.scatter(theta, phi_corr, label='Data Points', color='blue')
plt.plot(theta_fit_poly, phi_corr_poly_fit, label='Polynomial Fit', color='red')
plt.xlabel('Theta')
plt.ylabel('Phi_corr')
plt.title('Polynomial Fit for Phi_corr as a Function of Theta')
plt.legend()
plt.grid(True)
plt.show()

# Display polynomial coefficients for phi_corr as a function of theta
print(f"Polynomial Coefficients for Phi_corr as a Function of Theta: {polynomial_coefficients_phi_corr}")

