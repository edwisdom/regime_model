import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def utility(c, r, D):
	return (np.sinh(c)/np.sinh(D)) * (2/(1+np.exp(D - r)))

def utplot():
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Make data.
	X = np.arange(0, 3, 0.1)
	Y = np.arange(0, 5, 0.1)
	X, Y = np.meshgrid(X, Y)
	Z = utility(X, Y, 2.5)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	ax.set_zlim(0, 3)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.title("Utility Function")
	plt.xlabel("Capacity")
	plt.ylabel("Resources")
	plt.show()

utplot()

# dem = 1.0
# while dem:
# 	cap = float(input("Enter a capacity: "))
# 	res = float(input("Enter resources: "))
# 	dem = float(input("Enter a demand: "))
# 	print ("Utility: " + str(utility(cap, res, dem)))



