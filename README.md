## EasyMLSelector
```
    pip install EasyMLSelector
```
    * A class for training and testing many models found in sklearn.
    * Training and Testing for a large number of ML models in one go.
    * NOTE: Works best with type_filter="regressor"
    * Best Used with a small dataset to begin an investigation into the best models for your data.

### Don't know what kind of model is going to best suit your data?
```
    from EasyMLSelector import EasyMLSelector
```

### It's as easy as taking your X and y data and using these 2 commands
```
    M = EasyMLSelector(type_filter="regressor", Xy_tuple=(X,y))
    M.model_loop()
```

#### Then toy around with the best performing model:
```
    M.test_best()
```
```
    M.best_model
```