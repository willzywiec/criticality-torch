"""criticality -- modeling fissile material operations in nuclear facilities.

Python/torch port of the R ``criticality`` package
(https://github.com/willzywiec/criticality), based on
Zywiec et al. (2021) <doi:10.1016/j.ress.2020.107322>.

R -> Python function mapping:
    BN()        -> bn.build_bn()
    Tabulate()  -> tabulate.tabulate()
    Scale()     -> scale.scale_data()
    Model()     -> model.build_model()
    NN()        -> nn.train_nn()
    Test()      -> ensemble.test_ensemble()
    Plot()      -> plot.plot_history()
    Predict()   -> predict.predict_keff()
    Risk()      -> risk.estimate_risk()
"""

from .bn import BayesNet, build_bn
from .ensemble import test_ensemble
from .model import Metamodel, build_model, sse_loss
from .nn import train_nn
from .plot import plot_history
from .predict import predict_keff
from .risk import estimate_risk
from .scale import Dataset, scale_data
from .tabulate import tabulate

__all__ = [
    "BayesNet",
    "build_bn",
    "test_ensemble",
    "Metamodel",
    "build_model",
    "sse_loss",
    "train_nn",
    "plot_history",
    "predict_keff",
    "estimate_risk",
    "Dataset",
    "scale_data",
    "tabulate",
]

__version__ = "0.9.4"
