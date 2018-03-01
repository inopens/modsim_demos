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

def model(tijdstappen, init, varnames, f, returnDataFrame=False,
          plotresults=True, **kwargs):
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

    if (plotresults & (not returnDataFrame)): return modeloutput.plot()
    elif (plotresults & returnDataFrame): return modeloutput, modeloutput.plot()
    else: return modeloutput

def sensitiviteit(tijdstappen, init, varnames, f, parameternaam,
                  perturbatie=0.0001, soort='absoluut', **kwargs):
    """
    Berekent de gevoeligheidsfunctie(s) van de modeloutput(s) naar één bepaalde parameter

    Argumenten
    -----------
    tijdstappen: np.array
        array van tijdstappen

    init: list
        lijst met initiële condities

    varnames: list
        lijst van strings met namen van de variabelen

    f: function
        functie die de afgeleiden definieert die opgelost moeten worden

    parameternaam : string
        naam van de parameter waarvoor de gevoeligheidsfunctie moet opgesteld worden

    perturbatie: float
        perturbatie van de parameter

    kwargs: dict
        functie specifieke parameters
    """
    res_basis = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    parameterwaarde_basis = kwargs.pop(parameternaam)
    kwargs[parameternaam] = (1 + perturbatie) * parameterwaarde_basis
    res_hoog = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    kwargs[parameternaam] = (1 - perturbatie) * parameterwaarde_basis
    res_laag = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    if soort == 'absoluut':
        sens = (res_hoog - res_laag)/(2.*perturbatie*parameterwaarde_basis)

    if soort == 'relatief parameter':
            sens = (res_hoog - res_laag)/(2.*perturbatie*parameterwaarde_basis)*parameterwaarde_basis

    if soort == 'relatief variabele':
        sens = (res_hoog - res_laag)/(2.*perturbatie*parameterwaarde_basis)/res_basis

    if soort == 'relatief totaal':
        sens = (res_hoog - res_laag)/(2.*perturbatie*parameterwaarde_basis)*parameterwaarde_basis/res_basis

    return sens
