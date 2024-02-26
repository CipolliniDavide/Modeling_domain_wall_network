#! /usr/bin/env python3

import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# import nefi_short
import numpy as np
from matplotlib.lines import Line2D
from skimage import io
from skimage.color import gray2rgb, rgb2gray
from tap import Tap

from neuromorphic_materials.junction_graph_extraction.Class import nefi_short
from neuromorphic_materials.junction_graph_extraction.Class.graph_sheng_junc import (
    graph_sheng_junc,
    plot_graph,
)

# from Class.superpix_graph import superpix_graph


class BuildJunctionGraphParser(Tap):
    input_file: Path  # Input file
    output_dir: Path = Path("./data")  # Output directory path
    save_name: str = None  # Output file name
    junclets_executable: Path = Path(  # Path to junclets executable
        "./beyondOCR_junclets/my_junc"
    )

    def configure(self) -> None:
        self.add_argument("input_file")
        self.add_argument("-o", "--output_dir")
        self.add_argument("-s", "--save_name")


parser = BuildJunctionGraphParser()
args = parser.parse_args()
if args.save_name is None:
    args.save_name = args.input_file.stem

save_name = args.save_name
save_dir = args.output_dir / args.save_name
save_dir.mkdir(parents=True, exist_ok=True)

image = io.imread(args.input_file)

if len(image.shape) == 3 and image.shape[2] == 3:
    image = rgb2gray(image)

image = cv2.bitwise_not(image)

# Binary version of uint8 grayscale image
image_binary = (image / 255).astype(np.uint8)

ppm_image_path = (save_dir / save_name).with_suffix(".ppm")
cv2.imwrite(str(ppm_image_path.resolve()), gray2rgb(image))

# Call make junclets
label = "None"
outputfile = save_dir / save_name
binary = "0"
model = "1"
test = subprocess.Popen(
    [args.junclets_executable, ppm_image_path, label, outputfile, model, binary],
    stdout=subprocess.PIPE,
)
output = test.communicate()[0]

# Load Junc Coordinates
p_file = save_dir / f"{save_name}_points.txt"
with p_file.open() as f:
    lines = f.readlines()
coordinates_tuple = np.asarray(
    [(int(s) for s in line.split() if s.isdigit()) for line in lines]
)
coordinates = np.asarray(
    [[int(s) for s in line.split() if s.isdigit()] for line in lines]
)

plt.imshow(image, "gray")
plt.scatter(coordinates[:, 0], coordinates[:, 1], marker="+", c="red")
# plt.scatter(coordinates_refined[:,0], coordinates_refined[:,1], marker="+", c='yellow')
plt.savefig(save_dir / "nodes.png")
plt.close()

# Load Features of Junc
p_file = save_dir / f"{save_name}_features.txt"
with p_file.open() as f:
    lines = f.readlines()

l = []
for line in lines:
    temp = []
    for t in line.split():
        try:
            temp.append(float(t))
        except ValueError:
            pass
    l.append(temp)
features = np.asarray(l)

full_features = features
full_coordinates = coordinates

"""
def discard_2Junc_line(feat, thresh_overlap = 0.1 * np.pi):
    from scipy import signal
    peaks, prop = signal.find_peaks(feat, (None, None))
    sorted_peak = np.argsort(prop['peak_heights'])[::-1]
    angle = np.linspace(0, 360, num=len(feat), dtype=float) #* np.pi / 180.0
    plt.plot(feat)
    plt.plot(peaks[sorted_peak[:2]], feat[peaks[sorted_peak[:2]]], "x")
    plt.xticks(peaks[sorted_peak[:2]], ['%.2f' %an for an in angle[peaks[sorted_peak[:2]]]])
    plt.savefig('temp.png')
    plt.close()
    distance_peack = np.abs(angle[peaks[sorted_peak[0]]] - (angle[peaks[sorted_peak[1]]] + 180))  # degrees
    #distance_peack= np.abs(peaks[sorted_peak[0]]*bin_angle - (peaks[sorted_peak[1]]*bin_angle + np.pi)) #degrees
    print(distance_peack, angle[peaks[sorted_peak[0]]], angle[peaks[sorted_peak[1]]])
    if (np.deg2rad(distance_peack) < thresh_overlap) and (prop['peak_heights'][sorted_peak[2]]<0.01):
        return 1
    else: return 0

# Prune 2Junc on straight line
keep_ind_straight_2d= []
thresh_delta = 0.1 * np.pi
for count, feat in enumerate(features):
    mov_average = 10
    feat = np.convolve(feat, np.ones(mov_average) / mov_average, mode='valid')
    if discard_2Junc_line(feat, thresh_overlap = thresh_delta)==1: pass
    else: keep_ind_straight_2d.append(count)
print(len(keep_ind_straight_2d),'/', len(full_coordinates))
print(keep_ind_straight_2d)

from scipy.spatial.distance import cityblock
keep_ind= []
for count1 in range(len(coordinates)):
    for count2 in range(count1+1, len(coordinates)):
        if cityblock(coordinates[count1], coordinates[count2])<3 and cityblock(coordinates[count1], coordinates[count2])>0:
            print('eliminato ', count2, '.Tra ', count1, count2)
        else: keep_ind.append(count1)
coordinates_refined= coordinates[np.unique(keep_ind)]

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return dist_2 #np.argmin(dist_2)

dist_2= closest_node(full_coordinates[1], full_coordinates)

from Class.graph_sheng_junc import graph_sheng_junc, plot_graph, plot_graph_node_polarplot
save_dir_delta= save_dir + '/thresh_Delta%.3f/'%thresh_delta
utils.ensure_dir(save_dir_delta)
features=features[keep_ind_straight_2d]
coordinates= coordinates[keep_ind_straight_2d]
G= graph_sheng_junc(features=features, coordinates=coordinates, image=image, save_dir=save_dir_delta)
plot_graph(features=features, coordinates=coordinates, G_sheng= G, image= image, save_dir=save_dir_delta)
plot_graph_node_polarplot(features=features, coordinates=coordinates, G_sheng= G, image= image, save_dir=save_dir_delta)
"""

save_dir_full = save_dir / "full"
save_dir_full.mkdir(parents=True, exist_ok=True)
save_dir_full_str = f"{str(save_dir_full.resolve())}/"
G_full = graph_sheng_junc(
    features=full_features,
    coordinates=full_coordinates,
    image=image,
    binary_img=(1 - image_binary),
    save_fold=save_dir_full_str,
)
print(f"number of nodes: {len(G_full.nodes)}, edges: {len(G_full.edges)}")
plot_graph(
    features=full_features,
    coordinates=full_coordinates,
    G_sheng=G_full,
    image=image,
    save_fold=save_dir_full_str,
)
# Plot polar plots of individual junctions
# plot_graph_node_polarplot(
#     features=full_features,
#     coordinates=full_coordinates,
#     G_sheng=G_full,
#     image=image,
#     save_fold=save_dir_full_str,
# )
a = 0

# Extract graph with standard NEFI
otsued = nefi_short.otsu_process([image])["img"]
plt.imshow(otsued)
plt.savefig("otsu.png")
plt.close()
otsued = otsued // 255
skeleton = nefi_short.thinning([otsued])["skeleton"]
plt.imshow(skeleton, "gray")
plt.savefig("skeleton.png")
plt.close()
G_zhang = nefi_short.zhang_suen_node_detection(skeleton * 255)
G1 = nefi_short.breadth_first_edge_detection(skeleton, otsued, G_zhang)
print(f"number of nodes NEFI: {len(G_zhang.nodes)}, edges NEFI: {len(G_zhang.edges)}")
coord_zhang = np.asarray([[y, x] for x, y in G_zhang.nodes()])
fig = plt.figure()  # figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image, "gray")
ax.scatter(coord_zhang[:, 0], coord_zhang[:, 1], marker="+", c="red")
# ax.scatter(241, 27, marker="+", c='yellow')
# ax.scatter(223,73,  marker="+", c='yellow')
for n1, n2, attr in G_zhang.edges(data=True):
    l = Line2D(
        [n2[1], n1[1]], [n2[0], n1[0]], alpha=0.4
    )  # [0], n1[1], n2[0], n2[1], c=dic['weight'], alpha=0.5)
    ax.add_line(l)
# plt.scatter(coordinates_refined[:,0], coordinates_refined[:,1], marker="+", c='yellow')
plt.savefig("zhang_detect")
plt.close()

a = 0
