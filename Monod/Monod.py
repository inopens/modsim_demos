"""
Functies gebruikt voor vb Monod model

Ingmar Nopens
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

def model_afgeleiden(variables, t, mu_max, Q, V, Y, K_S, S_in):
 
    X = variables[0]
    S = variables[1]
    
    X_new = mu_max * S/(K_S + S) *X - Q/V*X
    S_new = - mu_max/Y * S/(K_S + S) *X + Q/V*(S_in - S)
    return [X_new, S_new]

def Monod_model(tijdstappen, X_0, S_0, mu_max, Q, V, Y, K_S, S_in, returnDataFrame=True):
    """
    Modelimplementatie van het Monod model 
    
    Parameters
    -----------
    tijdstappen : np.array
        array van tijdstappen          
    """
    modeloutput = odeint(model_afgeleiden, [X_0, S_0], tijdstappen, args=(mu_max, Q, V, Y, K_S, S_in))
    modeloutput = pd.DataFrame(modeloutput, columns=['X','S'], index=tijdstappen)
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