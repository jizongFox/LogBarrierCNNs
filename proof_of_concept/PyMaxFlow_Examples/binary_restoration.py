
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as ppl
from PIL import Image
import maxflow

img = imread("a2.png")
img = (img>0.5*255)*255

# Create the graph.
g = maxflow.Graph[int](0, 0)
# Add the nodes.
nodeids = g.add_grid_nodes(img.shape)
# Add edges with the same capacities.
g.add_grid_edges(nodeids, 500)
# Add the terminal edges.
g.add_grid_tedges(nodeids, img, 255-img)

graph = g.get_nx_graph()

# Find the maximum flow.
g.maxflow()
# Get the segments.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.
ppl.imshow(img2, cmap=ppl.cm.gray, interpolation='nearest')
ppl.show()
img2_ = Image.fromarray((img2*255).astype(np.uint8))
img2_.save('a2_gt.png',format='png')
