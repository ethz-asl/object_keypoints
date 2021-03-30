import numpy as np

def compute_bounding_box(mesh_file):
    from stl.mesh import Mesh
    mesh = Mesh.from_file(mesh_file)
    vertices = np.concatenate([mesh.v0, mesh.v1, mesh.v2], axis=0)
    min_x = vertices[:, 0].min()
    min_y = vertices[:, 1].min()
    min_z = vertices[:, 2].min()
    max_x = vertices[:, 0].max()
    max_y = vertices[:, 1].max()
    max_z = vertices[:, 2].max()
    minmax = np.array([[min_x, min_y, min_z],
        [max_x, max_y, max_z]])
    minimum = minmax[0]
    axes = np.array([[max_x - min_x, 0.0, 0.0],
        [0.0, max_y - min_y, 0.0],
        [0.0, 0.0, max_z - min_z]])
    bbox = np.array([minimum,
        minimum + axes[0],
        minimum + axes[1],
        minimum + axes[1] + axes[0],
        minimum + axes[2],
        minimum + axes[2] + axes[0],
        minimum + axes[2] + axes[1],
        minimum + axes[2] + axes[1] + axes[0]
    ])
    # returns 8 x 3 x 1 -> 8 column vectors of vertices.
    return np.concatenate([bbox, np.ones(8)], axis=1)[:, :, None]

