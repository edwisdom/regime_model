import pandas as pd 

import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from enum import Enum

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

PATH = "."

shock_strings = {'pos': 'with Positive Shock', 
				 'neg': 'with Negative Shock',
				 '': 'with No Shock'}

def create_fio(filename):
	file = filename[:-4] # Remove .pkl extension
	shock_tag = file[-3:]
	if shock_tag == 'neg' or shock_tag == 'pos':
		shock, network = (shock_strings[file[-3:]], file[:-3])
	else:
		shock, network = (shock_strings[''], file)
	network = (network.replace("_", " ")).title() + "Network "
	return (filename, ["Connectivity", "Capacity"],
			["Ratio Satisfied", "Average Clustering", 
			"Assortativity", "Number of Components"],
			network, shock)

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
	filename, inputs, outputs, network, shock = fio
	df = pd.read_pickle(filename)
	for output in outputs:
		draw_plot(df, inputs, output, network, shock)

read_files()
plt.show()
