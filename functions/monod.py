"""
Functies gebruikt voor vb Monod model

Ingmar Nopens, Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementatie  Monod model
# ----------------------------

def model_afgeleiden(variables, t, kwargs):

    X = variables[0]
    S = variables[1]

    X_new = kwargs['mu_max'] * S/(kwargs['K_S'] + S) * X - kwargs['Q']/kwargs['V']*X
    S_new = - kwargs['mu_max']/kwargs['Y'] * S/(kwargs['K_S'] + S) *X + kwargs['Q']/kwargs['V']*(kwargs['S_in'] - S)
    return [X_new, S_new]
