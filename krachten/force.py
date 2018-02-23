"""
Functies gebruikt voor vb force model

Ingmar Nopens
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

def model_afgeleiden(variables, t, b, m, k, Fex):
 
    x1 = variables[0]
    x2 = variables[1]
    
    x1_new = x2
    x2_new = -b/m*x2 - k/m*x1 + Fex
    return [x1_new, x2_new]

def force_model(tijdstappen, x1_0, x2_0, b, m, k, Fex, returnDataFrame=True):
    """
    Modelimplementatie van het Force model 
    
    Parameters
    -----------
    tijdstappen : np.array
        array van tijdstappen          
    """
    modeloutput = odeint(model_afgeleiden, [x1_0, x2_0], tijdstappen, args=(b, m, k, Fex))
    modeloutput = pd.DataFrame(modeloutput, columns=['x1','x2'], index=tijdstappen)
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