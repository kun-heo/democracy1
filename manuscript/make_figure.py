"""Rebuild the manuscript's figure from the verified equilibrium payoff functions."""
from math import sqrt
from fractions import Fraction as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from continuous_mandates import contest

BLUE = "#1D526F"
RED = "#A74E32"
GRAY = "#646D73"
LOW = F(31788, 209683)
HIGH = F(".402648")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#273138", "xtick.color": "#273138",
                         "ytick.color": "#273138"})
    fig, (mandates, office) = plt.subplots(1, 2, figsize=(7.7, 3.12),
                                           gridspec_kw={"width_ratios": [1.24, 1]})
    for ax in (mandates, office):
        ax.set_xlim(-.012, 1.012)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
        ax.set_xlabel("Current monitoring, $S$")
        ax.grid(axis="y", color="#e2e6e8", linewidth=.65)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#859096")
    x = np.linspace(.4, 1, 200)
    mandates.plot([0, .4, *x], [1, 1, *[sqrt(1+3.2/t)-2 for t in x]], ls="--", color=GRAY,
                  linewidth=1.7, label="One election")
    # Parametrization by the challenger's offer avoids a numerical optimizer.
    weight = [F(i, 220) for i in range(221)]
    c = F(".150993")
    monitor = [float(c / (contest(g, F(1), F(1))[0] + c)) for g in weight]
    mandates.plot([0, float(LOW)], [1, 1], color=BLUE, linewidth=2.5)
    mandates.plot(monitor, list(map(float, weight)), color=BLUE, linewidth=2.3, label="Two elections")
    mandates.plot([float(HIGH), 1], [1, 1], color=BLUE, linewidth=2.5)
    mandates.scatter([float(HIGH), float(HIGH)], [0, 1], c=[BLUE, "white"], edgecolors=BLUE,
                     zorder=4, s=27)
    mandates.set_ylim(-.045, 1.07)
    mandates.set_yticks([0,.5,1],["0",".5","1"])
    mandates.set_title("Challenger's mandate magnitude", loc="left", fontsize=10.5)
    mandates.legend(loc="upper center", bbox_to_anchor=(.5, -.31), ncol=2, frameon=False, fontsize=8.1, handlelength=1.65, columnspacing=.9)
    mid = float(HIGH)
    office.plot([0,1],[.45,.45],ls="--",color=GRAY,lw=1.8,label="Wins the vote")
    office.plot([0,mid],[.45,.45],color=RED,lw=2.5,label="Takes office")
    office.plot([mid,1],[.225,.225],color=RED,lw=2.5)
    office.scatter([mid,mid],[.45,.225],c=[RED,"white"],edgecolors=RED,zorder=4,s=26)
    office.set_ylim(0,.54)
    office.set_yticks([0,.225,.45],["0","22.5%","45%"])
    office.set_title("Challenger in the two-election game",loc="left",fontsize=10.5)
    office.legend(loc="upper center",bbox_to_anchor=(.5, -.31),ncol=2,frameon=False,fontsize=8.1,handlelength=1.6,columnspacing=.9)
    fig.subplots_adjust(left=.045,right=.993,bottom=.31,top=.87,wspace=.25)
    fig.savefig(Path(__file__).with_name("mandates_and_transfer.pdf"), bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()

