"""
Functies gebruikt voor vb nitrobenzeen

Ingmar Nopens, Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementatie  nitrobenzeen model
# ----------------------------

def model_afgeleiden(variables, t, kwargs):

    C6H6 = variables[0]
    N2O5 = variables[1]
    C6H5NO2 = variables[2]

    C6H6_new = -2*kwargs['k']*C6H6*C6H6*N2O5 + kwargs['Q']/kwargs['V']*(kwargs['C6H6_in'] - C6H6)
    N2O5_new = -kwargs['k']*C6H6*C6H6*N2O5 + kwargs['Q']/kwargs['V']*(kwargs['N2O5_in'] - N2O5)
    C6H5NO2_new = 2*kwargs['k']*C6H6*C6H6*N2O5 - kwargs['Q']/kwargs['V']*C6H5NO2
    return [C6H6_new, N2O5_new, C6H5NO2_new]
