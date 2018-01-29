import numpy as np
from run_model import *

inputs = np.array([[1.0, 1.5, 1.2, 1.3, 0.9, 1.5, 1.6],
                   [0.7, 1.1, 1.2, 1.7, 1.4, 0.5, 1.1]])

(X, ee, jac)  = lb_model3(inputs, plot=True)


