#! /usr/bin/env python3
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_spectral_entropy,
)


class PlotBfoEnsembleSeParser(Tap):
    bfo_graph_dir: Path = None  # Directory with BFO GraphML files
    beta_range: tuple[
        int, int, int
    ] = (  # Beta range in log10 space, last number is number of steps
        -3,
        4,
        300,
    )

    def configure(self) -> None:
        self.add_argument(nameof(self.bfo_graph_dir))


def create_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\beta$")
    ax.set_xscale("log")
    fig.set_size_inches(6, 3)
    return fig, ax


def main() -> None:
    args = PlotBfoEnsembleSeParser().parse_args()

    beta_range = np.logspace(*args.beta_range)
    bfo_graph_count = 0
    ensemble_se = np.zeros_like(beta_range)

    se_fig, se_ax = create_figure()
    se_ax.set_ylabel(r"$S(\mathbf{\rho}_{\beta})$")
    se_ax.set_title("Spectral entropy of BFO graphs")

    se_grad_fig, se_grad_ax = create_figure()
    se_grad_ax.set_ylabel(r"$\frac{S(\mathbf{\rho}_{\beta})}{\mathrm{d}\beta}$")
    se_grad_ax.set_title("Spectral entropy gradient of BFO graphs")

    for graph_file in sorted(args.bfo_graph_dir.glob("*.graphml")):
        bfo_graph_count += 1
        se = compute_spectral_entropy(nx.read_graphml(graph_file), beta_range)
        ensemble_se += se
        label = f"Crop ({graph_file.stem.replace('_', ', ')})"
        se_ax.plot(beta_range, se, label=label, linestyle="dotted")
        se_grad_ax.plot(
            beta_range, np.gradient(se, beta_range), label=label, linestyle="dotted"
        )

    ensemble_se /= bfo_graph_count

    fig, ax = create_figure()
    ax.plot(beta_range, ensemble_se, label="Ensemble")
    ax.set_ylabel(r"$S(\mathbf{\rho}_{\beta})$")
    ax.set_title("Spectral entropy of BFO ensemble")

    fig, ax = create_figure()
    ax.plot(beta_range, np.gradient(ensemble_se, beta_range), label="Ensemble")
    ax.set_ylabel(r"$\frac{S(\mathbf{\rho}_{\beta})}{\mathrm{d}\beta}$")
    ax.set_title("Spectral entropy gradient of BFO ensemble")

    plt.show()


if __name__ == "__main__":
    main()
