from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np

from model import RegimeModel
from model import num_satisfied, average_clustering
from model import assortativity, number_of_components


fixed_params = dict(num_nodes=20, demand=3, productivity=0.4, shape='small_world',
                    resource_inequality=0.5, uncertainty=0.2, shock=0.5)
variable_params = dict(network_param=np.linspace(0, 1, 51)[1:],
                       capacity_inequality=np.linspace(0,0.75,51)[1:])
model_reporter = {"Ratio Satisfied": lambda m: num_satisfied(m)/m.num_nodes,
                  "Average Clustering": lambda m: average_clustering(m),
                  "Assortativity": lambda m: assortativity(m),
                  "Number of Components": lambda m: number_of_components(m)}


class RegimeBatchRunner(BatchRunner):
    def run_model(self, model):
        model.run_model(self.max_steps)


def get_shock_string(value):
	if value > 1:
		shock = '_pos'
	elif value < 1:
		shock = '_neg'
	else:
		shock = ''


param_run = RegimeBatchRunner(RegimeModel, variable_parameters=variable_params,
                        fixed_parameters=fixed_params, model_reporters=model_reporter,
                        max_steps=10, display_progress=True)
param_run.run_all()
df = param_run.get_model_vars_dataframe()
shock = get_shock_string(fixed_params[shock])
df.to_pickle(fixed_params[shape] + shock + '.pkl')
