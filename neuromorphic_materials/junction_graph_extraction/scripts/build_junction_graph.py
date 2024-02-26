import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt

# import nefi_short
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from skimage import io
from skimage.color import gray2rgb, rgb2gray
from skimage.future import graph
from skimage.segmentation import slic

root_script = os.getcwd()
sys.path.extend([root_script])
# from Class.superpix_graph import add_attr_, add_attr_Cnn, remove_attribute
from Class.utils import utils

# from Class.superpix_graph import superpix_graph
from Class.visualize import visualize

input_fold = root_script.rsplit("/", 1)[0] + "/"
save_name = "cipollini_signature_inverted"  #'mini_JR38_elongation'
roothpath = input_fold
save_fold = roothpath + "junction_graph/{}/".format(save_name)
utils.ensure_dir(save_fold)

# save_name= 'firma'
# image= io.imread('/home/hp/Scaricati/junclets/firma.png')
image = rgb2gray(io.imread(input_fold + save_name + ".png"))
from skimage import morphology

# image= np.expand_dims(image, axis=2)
from skimage.filters import threshold_sauvola

image = utils.bilinear_resize_vectorized(image=image, newshape=(256, 256))
image = utils.scale(image, (0, 255)).astype("uint8")  # grayscale
image = cv2.bitwise_not(image)

window_size = (15, 15)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)
binary_sauvola = image > thresh_sauvola
from skimage.morphology import disk, square

binary_sauvola = morphology.binary_closing(binary_sauvola, square(2))
binary_sauvola = morphology.binary_erosion(binary_sauvola)
binary_file_name = save_fold + "sauvola"
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.savefig(binary_file_name + ".png")
cv2.imwrite(binary_file_name + ".png", (binary_sauvola * 255))
# Start subprocess
junclets_fold = "./beyondOCR_junclets/"
run_file_name = junclets_fold + "my_junc"
lib_list = [
    "beyondOCR_junclets.cpp",
    "dflJuncletslib.cpp",
    "dflPenWidth.cpp",
    "dflUtils.cpp",
    "pamImage.cpp",
    "dflBinarylib.cpp",
]
load_lib = [junclets_fold + i for i in lib_list]
# Create runnable
create_runnable = ["g++"] + load_lib + ["-pedantic", "-Wall", "-o", run_file_name]
test = subprocess.Popen(create_runnable, stdout=subprocess.PIPE)
output = test.communicate()[0]
# Convert to ppm
subprocess.Popen(
    ["convert", binary_file_name + ".png", binary_file_name + ".ppm"],
    stdout=subprocess.PIPE,
)
# Call make junclets
ppm_file = input_fold + save_name + ".ppm"
ppm_file = binary_file_name + ".ppm"
label = "None"
outputfile = save_fold + save_name
binary = str(0)
model = str(1)
test = subprocess.Popen(
    [run_file_name, ppm_file, label, outputfile, model, binary], stdout=subprocess.PIPE
)
output = test.communicate()[0]

# Load Junc Coordinates
p_file = save_fold + save_name + "_points.txt"
with open(p_file) as f:
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
plt.savefig(save_fold + "nodes.png")
plt.close()

# Load Features of Junc
p_file = save_fold + save_name + "_features.txt"
with open(p_file) as f:
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

from Class import nefi_short

# otsued= bin#nefi_short.otsu_process([image])['img']
# plt.imshow(otsued); plt.savefig(save_fold+'otsu.png'); plt.close()
# otsued=otsued//255
skeleton = nefi_short.thinning([binary_sauvola])["skeleton"]
plt.imshow(skeleton)
plt.savefig(save_fold + "skeleton.png")
plt.close()

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
save_fold_delta= save_fold + '/thresh_Delta%.3f/'%thresh_delta
utils.ensure_dir(save_fold_delta)
features=features[keep_ind_straight_2d]
coordinates= coordinates[keep_ind_straight_2d]
G= graph_sheng_junc(features=features, coordinates=coordinates, image=image, save_fold=save_fold_delta)
plot_graph(features=features, coordinates=coordinates, G_sheng= G, image= image, save_fold=save_fold_delta)
plot_graph_node_polarplot(features=features, coordinates=coordinates, G_sheng= G, image= image, save_fold=save_fold_delta)
"""
from Class.graph_sheng_junc import (
    graph_sheng_junc,
    plot_graph,
    plot_graph_node_polarplot,
)

save_fold_full = save_fold + "/full/"
utils.ensure_dir(save_fold_full)
G_full = graph_sheng_junc(
    features=full_features,
    coordinates=full_coordinates,
    image=image,
    binary_img=(binary_sauvola * 1).astype("uint8"),
    save_fold=save_fold_full,
)
plot_graph(
    features=full_features,
    coordinates=full_coordinates,
    G_sheng=G_full,
    image=image,
    save_fold=save_fold_full,
)
plot_graph_node_polarplot(
    features=full_features,
    coordinates=full_coordinates,
    G_sheng=G_full,
    image=image,
    save_fold=save_fold_full,
)
a = 0

"""
# Extract graph with standard NEFI
otsued= nefi_short.otsu_process([image])['img']
plt.imshow(otsued); plt.savefig('otsu.png'); plt.close()
otsued=otsued//255
skeleton= nefi_short.thinning([otsued])['skeleton']
plt.imshow(skeleton, 'gray'); plt.savefig('skeleton.png'); plt.close()
G_zhang= nefi_short.zhang_suen_node_detection(skeleton*255)
G1= nefi_short.breadth_first_edge_detection(skeleton, otsued, G_zhang)
coord_zhang= np.asarray([ [y, x] for x, y in G_zhang.nodes() ])
fig = plt.figure()#figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image, 'gray')
ax.scatter(coord_zhang[:,0], coord_zhang[:,1], marker="+", c='red')
#ax.scatter(241, 27, marker="+", c='yellow')
#ax.scatter(223,73,  marker="+", c='yellow')
for n1, n2, attr in G_zhang.edges(data=True):
    print(n1)
    l = Line2D([n2[1], n1[1]], [n2[0], n1[0]], alpha=.4)#[0], n1[1], n2[0], n2[1], c=dic['weight'], alpha=0.5)
    ax.add_line(l)
#plt.scatter(coordinates_refined[:,0], coordinates_refined[:,1], marker="+", c='yellow')
plt.savefig('zhang_detect')
plt.close()
"""
a = 0
