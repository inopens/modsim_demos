"""
algemene functies

Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from scipy.integrate import odeint
from scipy import optimize

base_context = {

    "font.size": 12,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "grid.linewidth": 1,
    "lines.linewidth": 4,#1.75,
    "patch.linewidth": .3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,

    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": .5,
    "ytick.minor.width": .5,

    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    }

context = 'notebook'
font_scale = 2

# Scale all the parameters by the same factor depending on the context
scaling = dict(paper=.8, notebook=1, talk=1.3, poster=1.6)[context]
context_dict = {k: v * scaling for k, v in base_context.items()}

# Now independently scale the fonts
font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
             "xtick.labelsize", "ytick.labelsize", "font.size"]
font_dict = {k: context_dict[k] * font_scale for k in font_keys}
context_dict.update(font_dict)

plt.rcParams.update(context_dict)

fivethirtyeight = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', fivethirtyeight)

figsize = (9,6)

def model(tijdstappen, init, varnames, f, returnDataFrame=False,
          plotresults=True, **kwargs):
    """
    Modelimplementatie

    Parameters
    -----------
    tijdstappen: np.array
        array van tijdstappen

    init: list
        lijst met initiele condities

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
    idx = pd.Index(data=tijdstappen, name='Tijd')
    modeloutput = pd.DataFrame(data, index=idx)

    if plotresults:
        fig, ax = plt.subplots(figsize=figsize)
        modeloutput.plot(ax=ax);
    if returnDataFrame:
        return modeloutput

def sensitiviteit(tijdstappen, init, varnames, f, parameternaam,
                  log_perturbatie=-4, soort='absoluut', **kwargs):
    """
    Berekent de gevoeligheidsfunctie(s) van de modeloutput(s) naar 1 bepaalde parameter

    Argumenten
    -----------
    tijdstappen: np.array
        array van tijdstappen

    init: list
        lijst met initiele condities

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
    perturbatie = 10**log_perturbatie
    res_basis = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    parameterwaarde_basis = kwargs.pop(parameternaam)
    kwargs[parameternaam] = (1 + perturbatie) * parameterwaarde_basis
    res_hoog = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    kwargs[parameternaam] = (1 - perturbatie) * parameterwaarde_basis
    res_laag = model(tijdstappen, init, varnames, f, returnDataFrame=True,
                     plotresults=False, **kwargs)
    if soort == 'absolute sensitiviteit':
        sens = (res_hoog - res_laag)/(2.*perturbatie)

    if soort == 'relatieve sensitiviteit parameter':
            sens = (res_hoog - res_laag)/(2.*perturbatie)*parameterwaarde_basis

    if soort == 'relatieve sensitiviteit variabele':
        sens = (res_hoog - res_laag)/(2.*perturbatie)/res_basis

    if soort == 'relatieve totale sensitiviteit':
        sens = (res_hoog - res_laag)/(2.*perturbatie)*parameterwaarde_basis/res_basis
    fig, ax = plt.subplots(figsize=figsize)
    sens.plot(ax=ax)
    ax.set_xlabel('Tijd')
    ax.set_ylabel(soort)

def sse(simulation, data):
    return np.sum(np.sum((np.atleast_2d(data) - np.atleast_2d(simulation))**2))

def track_calib(opt_fun, X, param_names, method='Nelder-Mead', tol=1e-4):
    """
    Optimalisatie m.b.v. het Nelder-Mead algoritme. Alle iteratiestappen worden bijgehouden.

    Argumenten
    ----------
    opt_fun : function
        optimalisatiefunctie
    X : list
        parameters
    method : str
        define the method for optimisation, options are: 'Neder-Mead', 'BFGS', 'Powell'
        'basinhopping', 'brute', 'differential evolution'
    tol: float
        tolerance to determine the endpoint of the optimisation. Is not used in
        method options 'brute' and 'basinhopping'

    Output
    ------
    parameters : DataFrame
        alle geteste parametercombinaties
    results : np.array
        alle bekomen objectiefwaarden

    """
    parameters = []
    results = []

    def internal_opt_fun(X):
        result = opt_fun(X)    # Bereken objectieffunctiewaarde voor de huidige set parameters
        parameters.append(X)   # Tussentijdse parameter waarden bijhouden
        results.append(result) # Tussentijdse SSE bijhouden
        return result

    if method in ['Nelder-Mead', 'BFGS', 'Powell']:
        res = optimize.minimize(internal_opt_fun, X, method=method, tol=tol)
    elif method == 'basinhopping':
        res = optimize.basinhopping(internal_opt_fun, X)
    elif method == 'brute':
        bounds = [(0.001*i, 100*i) for i in X]
        res = optimize.brute(internal_opt_fun, bounds)
    elif method == 'differential evolution':
        bounds = [(0.001*i, 100*i) for i in X]
        res = optimize.differential_evolution(internal_opt_fun, bounds, tol=tol)
    else:
        raise ValueError('use correct optimisation algorithm, see docstring for options')
    parameters = pd.DataFrame(np.array(parameters), columns=param_names)
    results = np.array(results)

    return parameters,results

def plot_calib(parameters, results, i, data, sim_model):
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(ax=ax, linestyle='', marker='.', markersize=15,
              colors=[fivethirtyeight[0], fivethirtyeight[1]])
    sim = sim_model(parameters.loc[i].values)
    sim.plot(ax=ax, linewidth=5,
             colors=[fivethirtyeight[0], fivethirtyeight[1]])
    ax.set_xlabel('Tijd')
    ax.set_ylabel('waarde variabelen');
    handles, labels = ax.get_legend_handles_labels()
    labels = [l+' simulation' if (i>= data.shape[1]) else l for i, l in enumerate(labels)]
    ax.legend(handles, labels)
    fig, ax = plt.subplots(figsize=figsize)
    cols = parameters.columns
    c = results - min(results)
    c *= 1/max(c)
    sc = ax.scatter(parameters[cols[0]], parameters[cols[1]], c=c, s=50, cmap="viridis", vmax=1)
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*max(c), 0.95*max(c)])
    cbar.set_ticklabels(['lage waarde\nobjectieffunctie', 'hoge waarde\nobjectieffunctie'])
    ax.scatter(parameters[cols[0]].iloc[0], parameters[cols[1]].iloc[0], marker='o', s=450, c=fivethirtyeight[2]) # startwaarde
    ax.scatter(parameters[cols[0]].iloc[-1], parameters[cols[1]].iloc[-1], marker='*', s=500, c=fivethirtyeight[1]) # eindwaarde
    ax.scatter(parameters[cols[0]].iloc[i], parameters[cols[1]].iloc[i], s=150, vmax=1, c=fivethirtyeight[4])        # huidige waarde
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_xlim(0.95*parameters[cols[0]].min(), 1.05*parameters[cols[0]].max())
    ax.set_ylim(0.95*parameters[cols[1]].min(), 1.05*parameters[cols[1]].max())

def plot_contour_monod(optimizer):
    n_points = 30
    mu_max = np.logspace(np.log10(0.001), np.log10(50), n_points)
    K_S = np.logspace(np.log10(0.001), np.log10(10), n_points)
    X_mu_max, X_K_S = np.meshgrid(mu_max, K_S)
    Z = np.array([optimizer(params) for params in zip(X_mu_max.flatten(), X_K_S.flatten())])
    Z = Z.reshape((n_points, n_points))
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.contourf(X_mu_max, X_K_S, Z, cmap='viridis')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
    cbar.set_ticklabels(['lage waarde\nobjectieffunctie', 'hoge waarde\nobjectieffunctie'])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('mu_max')
    ax.set_ylabel('K_S')

def plot_contour_force(optimizer):
    n_points = 30
    b = np.linspace(0, 2, n_points)
    k = np.linspace(0, 2, n_points)
    X_b, X_k = np.meshgrid(b, k)
    Z = np.array([optimizer(params) for params in zip(X_b.flatten(), X_k.flatten())])
    Z = np.log10(Z)
    Z = Z.reshape((n_points, n_points))
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.contourf(X_b, X_k, Z, cmap='viridis')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.05*Z.max(), 0.95*Z.max()])
    cbar.set_ticklabels(['lage waarde\nobjectieffunctie', 'hoge waarde\nobjectieffunctie'])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('b')
    ax.set_ylabel('k')