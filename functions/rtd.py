"""
Functies gebruikt voor vb Monod model

Daan Van Hauwermeiren
"""
# Importeren van functionaliteiten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

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

# ----------------------------
# Implementatie  RTD model
# ----------------------------

def expon_shifted(t, **kwargs):
    """
    $f:=$
    $0, \forall t < \tau_0$
    $1 - e^{-(x-\tau_0)/\beta}, \forall t \geq \tau_0$
    """
    y = 1 - np.exp(-(t-kwargs["tau_0"])/kwargs["beta"])
    y[t < kwargs["tau_0"]] = 0
    return y


def calculate_rtd(tijdstappen, model, returnDataFrame=False, plotresults=True,
                  **kwargs):
    """
    Modelimplementatie

    Parameters
    -----------
    tijdstappen: np.array
        array van tijdstappen

    model: function
        functie die de RTD definieert

    returnDataFrame: bool
        zet op True om de simulatiedata terug te krijgen

    kwargs: dict
        functie specifieke parameters
    """
    fvals = model(tijdstappen, **kwargs)
    modeloutput = pd.DataFrame(
        {"respons": fvals},
        index=pd.Index(data=tijdstappen, name='Tijd'))

    if plotresults:
        fig, ax = plt.subplots(figsize=figsize)
        modeloutput.plot(ax=ax);
        ax.set_ylim(-0.1, 1.1)
    if returnDataFrame:
        return modeloutput

def sensitiviteit(tijdstappen, model, parameternaam,
                  log_perturbatie=-4, soort='absoluut', **kwargs):
    """
    Berekent de gevoeligheidsfunctie(s) van de modeloutput(s) naar 1 bepaalde parameter

    Argumenten
    -----------
    tijdstappen: np.array
        array van tijdstappen

    model: function
        functie die de RTD definieert

    parameternaam : string
        naam van de parameter waarvoor de gevoeligheidsfunctie moet opgesteld worden

    perturbatie: float
        perturbatie van de parameter

    kwargs: dict
        functie specifieke parameters
    """
    perturbatie = 10**log_perturbatie
    res_basis = calculate_rtd(tijdstappen, model, returnDataFrame=True,
                     plotresults=False, **kwargs)
    parameterwaarde_basis = kwargs.pop(parameternaam)
    kwargs[parameternaam] = (1 + perturbatie) * parameterwaarde_basis
    res_hoog = calculate_rtd(tijdstappen, model, returnDataFrame=True,
                     plotresults=False, **kwargs)
    kwargs[parameternaam] = (1 - perturbatie) * parameterwaarde_basis
    res_laag = calculate_rtd(tijdstappen, model, returnDataFrame=True,
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


def plot_calib(parameters, results, i, data, sim_model):
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(ax=ax, linestyle='', marker='.', markersize=15,
              colors=[fivethirtyeight[0]])
    sim = sim_model(parameters.loc[i].values)
    sim.plot(ax=ax, linewidth=5,
             colors=[fivethirtyeight[1]])
    ax.set_xlabel('Tijd')
    ax.set_ylabel('waarde variabelen');
    handles, labels = ax.get_legend_handles_labels()
    labels = [l+' simulation' if (i>= data.shape[1]) else l for i, l in enumerate(labels)]
    ax.legend(handles, labels, loc='best')
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

def plot_contour_rtd(optimizer):
    n_points = 30
    beta = np.linspace(100, 500, n_points)
    tau_0 = np.linspace(100, 500, n_points)
    X_beta, X_tau_0 = np.meshgrid(beta, tau_0)
    Z = np.array([optimizer(params) for params in zip(X_beta.flatten(), X_tau_0.flatten())])
    Z = Z.reshape((n_points, n_points))
    lvls = np.logspace(math.log10(Z.min()), math.log10(Z.max()), 15)
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.contourf(X_beta, X_tau_0, Z, cmap='viridis', norm=LogNorm(), levels=lvls)
    ax.plot(206.003182, 373.223628, marker='o', color='white')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([1.1*Z.min(), 0.8*Z.max()])
    cbar.set_ticklabels(['lage waarde\nobjectieffunctie', 'hoge waarde\nobjectieffunctie'])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('beta')
    ax.set_ylabel('tau_0')
