
"""
How to use several calls to ``add_grid_edges`` and ``add_grid_tedges`` to
create a flow network with medium complexity.
"""

import numpy as np
import maxflow

from matplotlib import pyplot as plt
import networkx
from examples_utils import plot_graph_2d

def create_graph():
    np.random.seed(1)
    img = np.random.randint(0,10,size=(5,5))
    print(img)

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((5,5))


    kernel = np.ones((5,5))
    kernel[int(kernel.shape[0]/2),int(kernel.shape[1]/2)]=0
    # the kernel should be strictly square

    padding_size = int(max(kernel.shape)/2)
    position = np.array(list(zip(*np.where(kernel!=0))))

    def shift_matrix(matrix, kernel):
        center_x, center_y = int(kernel.shape[0]/2),int(kernel.shape[1]/2)
        [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel==1))))[0]
        dy, dx =  kernel_x -center_x, kernel_y - center_y
        shifted_matrix = np.roll(matrix, -dy, axis=0)
        shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
        return  shifted_matrix

    for p in position:
        structure  = np.zeros(kernel.shape)
        structure[p[0],p[1]] = kernel[p[0],p[1]]
        pad_im =  np.pad(img, ((padding_size,padding_size), (padding_size, padding_size)), 'constant', constant_values=0)
        shifted_im = shift_matrix(pad_im,structure)
        weights_ = np.abs(pad_im-shifted_im)[padding_size:-padding_size,padding_size:-padding_size]
        print(structure)
        print(weights_)
        g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=False)

    return nodeids, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    
    plot_graph_2d(g, nodeids.shape)


    nxgraph = g.get_nx_graph()
    A = networkx.adjacency_matrix(nxgraph).todense()
    print(np.allclose(A, A.T, atol=1e-10))
    fig, (ax1) = plt.subplots(nrows=1, figsize=(6, 10))
    ax1.imshow(A)
    plt.tight_layout()
    plt.show()
