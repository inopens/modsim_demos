"""
algemene functies

Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
#import seaborn as  sns
import numpy as np
import pandas as pd
import math

from scipy.integrate import odeint

def model(tijdstappen, init, varnames, f, returnDataFrame=False, **kwargs):
    """
    Modelimplementatie

    Parameters
    -----------
    tijdstappen: np.array
        array van tijdstappen

    init: list
        lijst met initiële condities

    varnames: list
        lijst van strings met namen van de variabelen

    f: function
        functie die de afgeleiden definieert die opgelost moeten worden

    returnDataFrame: bool
        zet op True om de simulatiedata terug te krijgen

    kwargs: dict
        functie specifieke parameters
    """
    fvals = odeint(f, init, tijdstappen, args=(kwargs,))
    data = {col:vals for (col, vals) in zip(varnames, fvals.T)}
    idx = pd.Index(data=tijdstappen, name='tijd')
    modeloutput = pd.DataFrame(data, index=idx)

    if returnDataFrame: return modeloutput, modeloutput.plot()
    else: return modeloutput.plot()
