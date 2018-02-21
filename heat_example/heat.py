"""
Functies gebruikt voor vb populatiemodel

Ingmar Nopens
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
import seaborn as  sns
import numpy as np
import pandas as pd
import math

from scipy.integrate import odeint

# ----------------------------
# Implementatie  heat model
# ----------------------------

def model_afgeleiden(variables, t, Ar, Ea, Q, V, rho, Cp, U, A, delta_rH, N2O5_in, N2O4_in, Tin, Tw):
 
    N2O5 = variables[0]
    N2O4 = variables[1]
    T = variables[2]
    
    R= 8.31
    N2O5_new = -2*Ar*math.exp(-Ea/(R*T))*N2O5 + Q/V*(N2O5_in - N2O5)
    N2O4_new = 2*Ar*math.exp(-Ea/(R*T))*N2O5 + Q/V*(N2O4_in - N2O4)
    T_new = 1./(V*rho*Cp)*(Q*rho*Cp*(Tin-T) + U*A*(Tw-T) -V*Ar*math.exp(-Ea/(R*T))*N2O5*delta_rH)
    return [N2O5_new, N2O4_new, T_new]

def heat_model(tijdstappen, N2O5_0, N2O4_0, T_0, Ar, Ea, Q, V, rho, Cp, U, A, delta_rH, N2O5_in, N2O4_in, Tin, Tw, returnDataFrame=True):
    """
    Modelimplementatie van het heat model 
    
    Parameters
    -----------
    tijdstappen : np.array
        array van tijdstappen          
    """
    modeloutput = odeint(model_afgeleiden, [N2O5_0, N2O4_0, T_0], tijdstappen, args=(Ar, Ea, Q, V, rho, Cp, U, A, delta_rH, N2O5_in, N2O4_in, Tin, Tw))
    modeloutput = pd.DataFrame(modeloutput, columns=['N2O5','N2O4','T'], index=tijdstappen)
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