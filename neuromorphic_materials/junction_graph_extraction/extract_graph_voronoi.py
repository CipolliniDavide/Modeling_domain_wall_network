import subprocess
import tempfile
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from skimage.color import gray2rgb
from skimage.morphology import skeletonize

from .Class import nefi_short


def extract_graph_from_binary_voronoi(
    image: np.ndarray, junclets_executable_path: Path
) -> nx.Graph | None:
    # Necessary assuming the image is an unaltered Voronoi sample image
    image = cv2.bitwise_not(image)

    with tempfile.TemporaryDirectory() as tmp_dir_path_str:
        tmp_dir = Path(tmp_dir_path_str)
        save_name = "junclets"
        ppm_image_path = tmp_dir / "image.ppm"
        cv2.imwrite(str(ppm_image_path.resolve()), gray2rgb(image))

        # Call make junclets
        subprocess.run(
            [
                junclets_executable_path,
                ppm_image_path,
                "None",
                tmp_dir / save_name,
                "1",
                "0",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Load Junc Coordinates
        p_file = tmp_dir / f"{save_name}_points.txt"
        with p_file.open() as f:
            lines = f.readlines()

    node_coordinates = sorted(
        tuple(int(s) for s in line.split() if s.isdigit()) for line in lines
    )

    graph_sheng = nx.Graph()
    # Add x and y coordinates as attributes, so they can be recovered from GraphML later
    #  Also invert coordinates tuple (y, x) so it works with NEFI's BF Edge Detection
    graph_sheng.add_nodes_from(
        (coords[::-1], dict(x=coords[0], y=coords[1])) for coords in node_coordinates
    )

    # Binary version of uint8 grayscale image
    image_binary = 1 - (image / 255).astype(np.uint8)
    skeleton = skeletonize(image_binary)
    connected_graph = nefi_short.breadth_first_edge_detection(
        skeleton, image_binary, graph_sheng
    )

    # Only return the largest connected component
    return connected_graph.subgraph(
        max(nx.connected_components(connected_graph), key=len)
    )
