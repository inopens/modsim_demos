"""
Functies gebruikt voor vb heat model

Ingmar Nopens, Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd
import math

from scipy.integrate import odeint

# ----------------------------
# Implementatie  heat model
# ----------------------------

def model_afgeleiden(variables, t, kwargs):

    N2O5 = variables[0]
    N2O4 = variables[1]
    T = variables[2]

    R = 8.31
    N2O5_new = -2*kwargs['Ar']*\
        math.exp(-kwargs['Ea']/(R*T))*N2O5 \
        + kwargs['Q']/kwargs['V']*(kwargs['N2O5_in'] - N2O5)
    N2O4_new = 2*kwargs['Ar']*math.exp(-kwargs['Ea']/(R*T))*N2O5 \
        + kwargs['Q']/kwargs['V']*(kwargs['N2O4_in'] - N2O4)
    T_new = 1./(kwargs['V']*kwargs['rho']*kwargs['Cp'])\
            *(kwargs['Q']*kwargs['rho']*kwargs['Cp']*(kwargs['Tin'] - T) \
            + kwargs['U']*kwargs['A']*(kwargs['Tw'] - T) \
            - kwargs['V']*kwargs['Ar']*math.exp(-kwargs['Ea']/(R*T))*N2O5*kwargs['delta_rH'])
    return [N2O5_new, N2O4_new, T_new]
