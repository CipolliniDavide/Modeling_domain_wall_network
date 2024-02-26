# Annotation instructions

This document contains instructions for setting up the environment, and explains how to run the programs necessary to annotate a BFO sample image.
The two programs required to fully annotate a sample are "Node Annotator" (`node_annotator`), and "Edge Annotator" (`edge_annotator`).

## Setup

1. Install Python 3.11 from the Microsoft store or [this link](https://www.python.org/downloads/release/python-3113/). On Linux (assuming Ubuntu and friends), see [this askubuntu answer](https://askubuntu.com/a/682875). For other distros, ask your favourite search engine.
2. Install dependencies by running `python3.11 -m pip install -r requirements.txt`
3. Run the programs using `python3.11 <program_name>`, or `<program_name>` if `python3` points to Python 3.11 on your system.

**Note:** to compile the junction detector 
In order to make sure the junction detector is compiled for your processor (Linux/Windows/MacOs).
Go into the folder neuromorphic_materials/junction_graph_extraction and execute:
`cd neuromorphic_materials/junction_graph_extraction`
`python compile_beyond_ocr.py`

**Note:** The programs might run on older versions of Python, but support is not guaranteed.

## Annotation

To annotate a sample, execute the following steps:

1. Run the Node Annotator (`node_annotator`)
2. Open the sample file
3. Annotate nodes
4. Save the result to a file
5. Run the Edge Annotator (`edge_annotator`)
6. Open the sample file and the nodes file you just saved
7. Annotate edges
8. Save the result to a file

Please keep save-file naming consistent with the sample files you received.

If the sun or another bright light source is shining directly on the screen, it can be difficult to see where nodes or edges should be placed.

## Program instructions

Both programs have three buttons and a sample area. The buttons are fairly straightforward.

* "**Open sample**" allows you to select (a) file(s) to open.
* "**Save {nodes/edges}**" will save the nodes or edges you placed. Loading incomplete annotations is currently not possible. Please finish annotating a sample image before saving it.
* "**Quit**" will ask for confirmation to stop the program, without saving.

The actual annotating of samples happens in the sample area. The controls are slightly different for each program. 

### Node Annotator

* **Left-click** on empty space to place a new node
* **Left-click** on a node to select it, turning it green
* Press **Delete** (not backspace) while a node is selected to remove it
* **Left-click** on empty space, or press **Escape** while a node is selected, to deselect
* **Left-click** on a selected node to deselect it

### Edge Annotator

* **Left-click** on a node to select it, turning it green
* **Left-click** on another node while a node is selected to create a new edge, and deselect the initial node
* **Left-click** on an edge to select it, turning it green
* Press **Delete** (not backspace) while an edge is selected to remove it
* **Left-click** on empty space, or press **Escape** while a node or edge is selected, to deselect it
* **Left-click** on a selected node or edge to deselect it

Sometimes, it can be tricky to select an edge. Try clicking on different parts of the edge, especially closer to one of the nodes it connects.

## Convert annotation to graphs

Run `python annotation_to_graph.py -h` for info on the order of paths to be given to load and save graphs.