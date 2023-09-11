import numpy as np
import matplotlib.pyplot as plt

# Data
iterations = np.array([30000, 45000, 60000, 75000, 90000])
train_error = np.array([148.69, 138.71, 123.55, 112.54, 107.87])
test_error = np.array([80.49, 67.69, 56.86, 49.38, 53.17])
train_error_pcutoff = np.array([40.08, 37.55, 36.49, 39.23, 44.21])
test_error_pcutoff = np.array([13.14, 14.45, 16.07, 16.83, 14.97])

# Plot train and test error without p-cutoff
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_error, 'o-', label='Train Error')
plt.plot(iterations, test_error, 's-', label='Test Error')

# Polynomial curve fitting (let's use degree=2 for quadratic fit)
z_train = np.polyfit(iterations, train_error, 2)
z_test = np.polyfit(iterations, test_error, 2)
p_train = np.poly1d(z_train)
p_test = np.poly1d(z_test)
plt.plot(iterations, p_train(iterations), '--', label='Train Error Trend')
plt.plot(iterations, p_test(iterations), '--', label='Test Error Trend')

plt.title('Train and Test Error vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot train and test error with p-cutoff
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_error_pcutoff, 'o-', label='Train Error w/ p-cutoff')
plt.plot(iterations, test_error_pcutoff, 's-', label='Test Error w/ p-cutoff')

# Polynomial curve fitting
z_train_pcutoff = np.polyfit(iterations, train_error_pcutoff, 2)
z_test_pcutoff = np.polyfit(iterations, test_error_pcutoff, 2)
p_train_pcutoff = np.poly1d(z_train_pcutoff)
p_test_pcutoff = np.poly1d(z_test_pcutoff)
plt.plot(iterations, p_train_pcutoff(iterations), '--', label='Train Error w/ p-cutoff Trend')
plt.plot(iterations, p_test_pcutoff(iterations), '--', label='Test Error w/ p-cutoff Trend')

plt.title('Train and Test Error with p-cutoff vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Error w/ p-cutoff')
plt.legend()
plt.grid(True)
plt.show()
