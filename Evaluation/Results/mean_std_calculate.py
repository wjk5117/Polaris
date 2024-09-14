import numpy as np


# Positional and angular error of the default tag experiments
mean_dis = [1.421, 1.484, 1.769, 1.393, 1.116, 2.977, 1.290, 2.674, 1.349, 1.325]
angular = [-1.224, -0.488, 0.084, -0.596, 2.308, -0.830, 2.849, -0.830, 0.109, 2.084]
angular_abs = [abs(i) for i in angular]
print("Default tag experiment")
print("(mean, std) of positional error: ", np.mean(mean_dis), np.std(mean_dis))
print("(mean, std) of angular error: ", np.mean(angular_abs), np.std(angular_abs))


# Positional and angular error of varying polarty orientations
mean_dis = [1.779, 2.338, 1.923, 1.658, 3.438, 2.846, 3.180, 2.744, 2.675, 4.528]
angular = [-1.627, -2.875, -2.122, -1.627, -1.548, -0.725, 0.313, -2.393, 1.626, -2.902]
angular_abs = [abs(i) for i in angular]
print("Variying polarity orientations experiment")
print("(mean, std) of positional error: ", np.mean(mean_dis), np.std(mean_dis))
print("(mean, std) of angular error: ", np.mean(angular_abs), np.std(angular_abs))



# Positional and angular error of iron fillings experiment
mean_dis= [3.457, 1.944, 2.867, 2.866, 0.987, 2.837, 4.114, 1.620, 1.211, 1.632]
angular = [-1.847, -1.371, -2.915, -1.198, 0.109, -0.460, -3.477, -0.830, 0.127, 0.451]
angular_abs = [abs(i) for i in angular]
print("Iron fillings experiment")
print("(mean, std) of positional error: ", np.mean(mean_dis), np.std(mean_dis))
print("(mean, std) of angular error: ", np.mean(angular_abs), np.std(angular_abs))
