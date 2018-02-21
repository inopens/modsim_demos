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
# Implementatie  populatie model
# ----------------------------

def model_afgeleiden(variables, t, r_v, K_v, K_m, d_nv, d_n1, d_n2, r_1, r_2, alpha_1, alpha_2, m2_in):

    v = variables[0]
    m1 = variables[1]
    m2 = variables[2]
    
    v_new = r_v*v*(1-v/K_v) - d_nv*v
    m1_new = r_1*m1*(1-(m1+m2)/K_m)-alpha_1*v*m1-d_n1*m1
    m2_new = r_2*m2*(1-(m1+m2)/K_m)-alpha_2*v*m2-d_n2*m2+m2_in
    return [v_new, m1_new, m2_new]

def populatie_model(tijdstappen, v_0, m1_0, m2_0, r_v, K_v, K_m, d_nv, d_n1, d_n2, r_1, r_2, alpha_1, alpha_2, m2_in, returnDataFrame=True, plotFig=True):
    """
    Modelimplementatie van het populatiemodel 
    
    Parameters
    -----------
    tijdstappen : np.array
        array van tijdstappen          
    """
    modeloutput = odeint(model_afgeleiden, [v_0, m1_0, m2_0], tijdstappen, args=(r_v, K_v, K_m, d_nv, d_n1, d_n2, r_1, r_2, alpha_1, alpha_2, m2_in));
    modeloutput = pd.DataFrame(modeloutput, columns=['v','m1','m2'], index=tijdstappen)
    if plotFig:
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
def abs_sensitiviteit(tijd, model, parameternaam, pert, args):
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

    perturbatie=pert
    
    parameterwaarde = args[parameternaam]
    args[parameternaam]= parameterwaarde + parameterwaarde*perturbatie
    populatie_hoog = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False);
    args[parameternaam]= parameterwaarde - parameterwaarde*perturbatie
    populatie_laag = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False);
    
    sens = (populatie_hoog - populatie_laag)/(2.*perturbatie*parameterwaarde)
    
    return sens
  
    
    
def rel_sensitiviteit_par(tijd, model, parameternaam, pert, args):
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

    perturbatie=pert
    
    parameterwaarde = args[parameternaam]
    args[parameternaam]= parameterwaarde + parameterwaarde*perturbatie
    populatie_hoog = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    args[parameternaam]= parameterwaarde - parameterwaarde*perturbatie
    populatie_laag = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    
    sens = (populatie_hoog - populatie_laag)/(2.*perturbatie*parameterwaarde)*(parameterwaarde)
    
    return sens 
    
def rel_sensitiviteit_var(tijd, model, parameternaam, pert, args):
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

    perturbatie=pert
    
    parameterwaarde = args[parameternaam]
    args[parameternaam]= parameterwaarde
    populatie_gewoon = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    args[parameternaam]= parameterwaarde + parameterwaarde*perturbatie
    populatie_hoog = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    args[parameternaam]= parameterwaarde - parameterwaarde*perturbatie
    populatie_laag = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    
    sens = (populatie_hoog - populatie_laag)/(2.*perturbatie*parameterwaarde)/(populatie_gewoon)
    
    return sens  
    
def rel_sensitiviteit_tot(tijd, model, parameternaam, pert, args):
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

    perturbatie=pert
    
    parameterwaarde = args[parameternaam]
    args[parameternaam]= parameterwaarde
    populatie_gewoon = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    args[parameternaam]= parameterwaarde + parameterwaarde*perturbatie
    populatie_hoog = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    args[parameternaam]= parameterwaarde - parameterwaarde*perturbatie
    populatie_laag = model(tijd,args['v_0'],args['m1_0'],args['m2_0'],args['r_v'],args['K_v'],args['K_m'],args['d_nv'],args['d_n1'],args['d_n2'],args['r_1'],args['r_2'],args['alpha_1'],args['alpha_2'], args['m2_in'], plotFig=False)
    
    sens = (populatie_hoog - populatie_laag)/(2.*perturbatie*parameterwaarde)*(parameterwaarde/populatie_gewoon)
    
    return sens        