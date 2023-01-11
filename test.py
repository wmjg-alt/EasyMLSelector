from EasyMLSelector import EasyMLSelector
import numpy as np
s = 500

X = np.random.rand(s,10)
y = np.log(X).mean(axis=1)
M = EasyMLSelector(type_filter="regressor", Xy_tuple=(X,y))
M.model_loop()
M.test_best()