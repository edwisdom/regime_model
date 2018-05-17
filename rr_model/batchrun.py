from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np

from new_model import RegimeModel
from new_model import num_satisfied, average_clustering
from new_model import assortativity, number_of_components


fixed_params = dict(num_nodes=20, demand=3, productivity=0.4, shape='ws',
                    resource_inequality=0.5, uncertainty=0.2)
variable_params = dict(network_parameter=np.linspace(0, 1, 51)[1:],
                       capacity_inequality=np.linspace(0,0.75,51)[1:])
model_reporter = {"Ratio Satisfied": lambda m: num_satisfied(m)/m.num_nodes,
                  "Average Clustering": lambda m: average_clustering(m),
                  "Assortativity": lambda m: assortativity(m),
                  "Number of Components": lambda m: number_of_components(m)}


class RegimeBatchRunner(BatchRunner):
    def run_model(self, model):
        model.run_model(self.max_steps)


param_run = RegimeBatchRunner(RegimeModel, variable_parameters=variable_params,
                        fixed_parameters=fixed_params, model_reporters=model_reporter,
                        max_steps=10, display_progress=True)
param_run.run_all()
df = param_run.get_model_vars_dataframe()
df.to_pickle("small_world_test.pkl")
