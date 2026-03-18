#%%
import numpy as np
from seasushi.tools import plot_taylor

ref_data = np.random.rand(100)
model1_data = np.random.rand(100)
model2_data = np.random.rand(100)
model3_data = np.random.rand(100)
models_data = [model1_data, model2_data, model3_data]
output_taylor_diagram = "taylor_diagram.png"
plot_taylor(ref_data, models_data, output_taylor_diagram)

# %%
