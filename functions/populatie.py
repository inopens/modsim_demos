"""
Functies gebruikt voor vb populatiemodel

Ingmar Nopens, Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementatie  populatie model
# ----------------------------

def model_afgeleiden(variables, t, kwargs):

    v = variables[0]
    m1 = variables[1]
    m2 = variables[2]

    v_new = kwargs['r_v']*v*(1-v/kwargs['K_v']) - kwargs['d_nv']*v
    m1_new = kwargs['r_1']*m1*(1-(m1+m2)/kwargs['K_m'])-kwargs['alpha_1']*v*m1-kwargs['d_n1']*m1
    m2_new = kwargs['r_2']*m2*(1-(m1+m2)/kwargs['K_m'])-kwargs['alpha_2']*v*m2-kwargs['d_n2']*m2+kwargs['m2_in']
    return [v_new, m1_new, m2_new]
