# ml/drift_detector.py
import numpy as np, torch
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from ml.model import MLP
from shared.utils import calculate_accuracy_within_threshold

def gridsearch_retrain(Xp, y, model_old, in_dim):
    net = NeuralNetRegressor(
        module      = MLP,
        module__in_dim = in_dim,
        max_epochs  = 10,
        lr          = 0.01,
        batch_size  = 8,
        optimizer   = torch.optim.Adam,
        criterion   = torch.nn.MSELoss,
        callbacks   = [EarlyStopping(patience=10)],
        verbose     = 0)

    param = {
        "lr": [0.1,0.01,0.001],
        "module__h1":[16,32,64],
        "module__h2":[8,16,32],
        "batch_size":[8,16,32],
    }
    gs = GridSearchCV(net, param, cv=5, n_jobs=-1,
                      scoring="neg_mean_squared_error", verbose=2)
    gs.fit(Xp, y)
    return gs.best_estimator_.module_
