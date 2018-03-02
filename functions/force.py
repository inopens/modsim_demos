"""
Functies gebruikt voor vb force model

Ingmar Nopens, Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementatie  Force model
# ----------------------------

def model_afgeleiden(variables, t, kwargs):

    x1 = variables[0]
    x2 = variables[1]

    x1_new = x2
    x2_new = -kwargs['b']/kwargs['m']*x2 - kwargs['k']/kwargs['m']*x1 + kwargs['Fex']
    return [x1_new, x2_new]
