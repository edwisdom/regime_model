import pandas as pd 

import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

PATH = "."

fio_1 = ("network_and_p_longer.pkl", 
		["Productivity", "Connectivity"],
		["Ratio Satisfied", "Gini Capacity", "Gini Resources"])

def create_fio(filename):
	file = filename[:-4] # Remove .pkl extension
	is_neg, network = (True, file[:-3]) if file[-3:] == 'neg' else (False, file)
	network = (network.replace("_", " ")).title() + " Network"
	return (filename, ["Connectivity", "Capacity"],
			["Ratio Satisfied", "Average Clustering", 
			"Assortativity", "Number of Components"],
			network, is_neg)

def get_measure(df, X, Y, measure):
	return np.array(df[measure]).reshape(X.shape[0], Y.shape[0])

def draw_plot(df, inputs, output, network, shock):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Make data.
	X = np.linspace(0, 1, 51)[1:]
	Y = np.linspace(0, 1, 51)[1:]
	X, Y = np.meshgrid(X, Y)
	Z = get_measure(df, X, Y, output)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.title(output + " in " + network + shock)
	plt.xlabel(inputs[0])
	plt.ylabel(inputs[1])
	plt.tight_layout()

def read_files():
	for filename in os.listdir(PATH):
		if filename.endswith(".pkl"):
			(unpack_fio(create_fio(filename)))

def unpack_fio(fio):
	filename, inputs, outputs, network, is_neg = fio
	shock = " with Negative Shock" if is_neg else " with No Shock"
	df = pd.read_pickle(filename)
	for output in outputs:
		draw_plot(df, inputs, output, network, shock)

read_files()
plt.show()