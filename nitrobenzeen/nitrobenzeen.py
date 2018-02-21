"""
Functies gebruikt voor vb populatiemodel

Ingmar Nopens
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
import seaborn as  sns
import numpy as np
import pandas as pd

from scipy.integrate import odeint

# ----------------------------
# Implementatie  nitrobenzeen model
# ----------------------------

def model_afgeleiden(variables, t, k, Q, V, C6H6_in, N2O5_in):
 
    C6H6 = variables[0]
    N2O5 = variables[1]
    C6H5NO2 = variables[2]
    
    C6H6_new = -2*k*C6H6*C6H6*N2O5 + Q/V*(C6H6_in - C6H6)
    N2O5_new = -k*C6H6*C6H6*N2O5 + Q/V*(N2O5_in - N2O5)
    C6H5NO2_new = 2*k*C6H6*C6H6*N2O5 - Q/V*C6H5NO2
    return [C6H6_new, N2O5_new, C6H5NO2_new]

def nitrobenzeen_model(tijdstappen, C6H6_0, N2O5_0, C6H5NO2_0, k, Q, V, C6H6_in, N2O5_in, returnDataFrame=True):
    """
    Modelimplementatie van het nitrobenzeenmodel 
    
    Parameters
    -----------
    tijdstappen : np.array
        array van tijdstappen          
    """
    modeloutput = odeint(model_afgeleiden, [C6H6_0, N2O5_0, C6H5NO2_0], tijdstappen, args=(k, Q, V, C6H6_in, N2O5_in))
    modeloutput = pd.DataFrame(modeloutput, columns=['C6H6','N2O5','C6H5NO2'], index=tijdstappen)
    modeloutput.plot()
    if returnDataFrame:
        return modeloutput    

# ----------------------------
# Model optimalisatie functies
# ----------------------------
def SSE(gemeten, model):
    """
    Functie om de Sum of Squared Errors (SSE) te berekenen tussen een set van gemeten en gemodelleerde waarden.

    Parameters
    -----------
    gemeten : np.array
        numpy array van lengte N, met de gemeten waarden
    model : np.array
        numpy array van lengte N, met de gemodelleerde waarden

    Notes
    -----
    We werken hier niet met de DataFrame-structuur, maar gewoon met 2 getallenreeksen. De gebruiker is verantwoordelijk dat
    de gemeten en gemodelleerde waardne overeenstemmen
    """
    residuals = gemeten.flatten() - model.flatten()
    return np.sum(residuals**2)

# ------------------------------
# Model sensitiviteiten functies
# ------------------------------
def sensitiviteit(tijd, model, parameternaam, args):
    """
    Berekent de gevoeligheidsfunctie(s) van de modeloutput(s) naar één bepaalde parameter
    
    Argumenten
    -----------
    tijd : np.array
        array van tijdstappen
    model : function (geen stringnotatie!)
        naam van het model
    parameternaam : string
        naam van de parameter waarvoor de gevoeligheidsfunctie moet opgesteld worden
    args : dictionairy
        alle parameterwaarden die het model nodig heeft
    """

    perturbatie=0.0001
    
    parameterwaarde = args[parameternaam]
    args[parameternaam]= parameterwaarde + parameterwaarde*perturbatie
    bier_hoog = model(tijd,args['X_0'],args['G_0'],args['E_0'],args['mu_max'],args['K_G'],args['k1'],args['k2'])
    args[parameternaam]= parameterwaarde - parameterwaarde*perturbatie
    bier_laag = model(tijd,args['X_0'],args['G_0'],args['E_0'],args['mu_max'],args['K_G'],args['k1'],args['k2'])
    
    sens = (bier_hoog - bier_laag)/(2.*perturbatie*parameterwaarde)
    
    return sens