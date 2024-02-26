import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform

from .utils import utils


def polar_plot(
    data,
    mask=None,
    fig=None,
    geometry=(111),
    fontdict={"font_size_xlabel": 30, "font_size_ylabel": 25, "fontweight": "bold"},
    figsize=(10, 10),
    color="blue",
    mov_average=1,
    save_name=None,
    save_fold="./",
    title=None,
    suptitle=None,
):
    """
    :param data: array with pdf of each angle( degree )
    :param fig:
    :param geometry:
    :param fontdict:
    :param figsize:
    :param mov_average:
    :param save_name:
    :param save_fold:
    :param title:
    :return:
    """
    # data= np.deg2rad(data)
    # Create bins of angles and convert to radians
    angle = np.linspace(0, 360, num=len(data), dtype=float) * np.pi / 180.0
    # data = np.convolve(sorted(list(data)), np.ones(mov_average) / mov_average, mode='valid')
    # lux = [np.radians(a) for a in angle]
    # h, b= np.histogram(lux, bins=angle)
    # h, _, b = utils.empirical_pdf_and_cdf(lux, bins=angle)
    h = data
    # plt.clf()
    if fig is None:
        fig = plt.figure(figsize=figsize)
    sp = fig.add_subplot(geometry, projection="polar")  # Remove x and y ticks
    # sp = plt.subplot((111), projection='polar')
    sp.plot(angle, h, linewidth=6, alpha=0.6, c=color)
    if mask is None:
        pass
    else:
        sp.plot(angle[mask], h[mask], linewidth=6, alpha=0.6, c="red")
    sp.set_theta_zero_location("E")
    sp.set_theta_direction(-1)

    font_size_xlabel = fontdict["font_size_xlabel"]
    font_size_ylabel = fontdict["font_size_ylabel"]
    font_weight = fontdict["fontweight"]
    yticks = np.linspace(h.min(), h.max(), 3)
    sp.set_yticks(yticks)
    sp.set_yticklabels(["%.6f" % s for s in yticks])
    sp.tick_params(axis="x", labelsize=font_size_xlabel)
    sp.tick_params(axis="y", labelsize=font_size_ylabel)
    sp.set_title(title, weight="bold", fontsize="xx-large")
    plt.suptitle(suptitle, fontweight=font_weight, fontsize=font_size_xlabel)
    plt.tight_layout()
    if save_name:
        utils.ensure_dir(save_fold)
        plt.savefig(save_fold + save_name + "_polar_angle.png")
        plt.close()
    # else:
    # plt.show()
    # return [fig, sp]
    # plt.close()
