from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import utils


def polar_plot(
    data,
    mask=None,
    fig=None,
    geometry=111,
    font_dict=None,
    fig_size=(10, 10),
    color="blue",
    save_filename=None,
    save_dir: Path = Path.cwd(),
    title=None,
    suptitle=None,
    show=False,
):
    if font_dict is None:
        font_dict = {
            "font_size_x_label": 30,
            "font_size_y_label": 25,
            "font_weight": "bold",
        }

    # Create bins of angles and convert to radians
    angle = (
        np.linspace(0, 360, num=len(data), endpoint=False, dtype=float) * np.pi / 180.0
    )
    # If no existing figure is passed, create a new one
    if fig is None:
        fig = plt.figure(figsize=fig_size)
    sp = fig.add_subplot(geometry, projection="polar")  # Remove x and y ticks
    # sp = plt.subplot((111), projection='polar')
    sp.plot(angle, data, linewidth=6, alpha=0.6, c=color)
    if mask is None:
        pass
    else:
        sp.plot(angle[mask], data[mask], linewidth=6, alpha=0.6, c="red")
    sp.set_theta_zero_location("E")
    sp.set_theta_direction(1)

    font_size_x_label = font_dict["font_size_x_label"]
    font_size_y_label = font_dict["font_size_y_label"]
    font_weight = font_dict["font_weight"]
    y_ticks = np.linspace(data.min(), data.max(), 3)
    sp.set_yticks(y_ticks)
    sp.set_yticklabels(["%.6f" % s for s in y_ticks])
    sp.tick_params(axis="x", labelsize=font_size_x_label)
    sp.tick_params(axis="y", labelsize=font_size_y_label)
    sp.set_title(title, weight="bold", fontsize="xx-large")
    plt.suptitle(suptitle, fontweight=font_weight, fontsize=font_size_x_label)
    plt.tight_layout()
    if show:
        plt.show()
    elif save_filename is not None:
        utils.ensure_dir(save_dir)
        fig.savefig(save_dir / f"{save_filename}_polar_angle.png")
        plt.close()
