# SpConv: PyTorch Spatially Sparse Convolution Library

Some comments about the usage of the library

### Difference between [https://github.com/traveller59/spconv](https://github.com/traveller59/spconv)

- fix bug in `points_to_voxel_3d_np_mean` according to [https://github.com/traveller59/spconv/issues/135][https://github.com/traveller59/spconv/issues/135]


## Usage

- `SparseConv2d` is `SC` in the original paper `3D Semantic Segmentation with Submanifold Sparse Convolutional Networks`
- `SubMConv2d` is `SSC` in the original paper `3D Semantic Segmentation with Submanifold Sparse Convolutional Networks`
- `SparseConvTranspose2d` is the inverse operation of `SparseConv2d`. It is suggested not to use it since it will generate too much new points according to [https://github.com/traveller59/spconv/issues/149#issuecomment-639351324](https://github.com/traveller59/spconv/issues/149#issuecomment-639351324)
- `SparseInverseConv2d`

### SparseConvTensor

```Python
features = # your features with shape [N, numPlanes]
indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
batch_size = # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
print(x.sparity) # helper function to check sparity. 
```

### Sparse Convolution

```Python
import spconv
from torch import nn
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)# .dense()
```

### Inverse Convolution

Inverse sparse convolution means "inv" of sparse convolution. the output of inverse convolution contains same indices as input of sparse convolution.

Inverse convolution usually used in semantic segmentation.

```Python
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, 2, indice_key="cp0"),
            spconv.SparseInverseConv3d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)
```

### Utility functions

* convert point cloud to voxel

```Python

voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-50, -50, -3, 50, 50, 1],
    max_num_points=30,
    max_voxels=40000
)

points = # [N, 3+] tensor.
voxels, coords, num_points_per_voxel = voxel_generator.generate(points)


import spconv
import numpy as np

voxel_generator = spconv.utils.VoxelGenerator(
        voxel_size=[0.2, 0.2, 0.2],
        point_cloud_range= [0, 0, 0, 1, 1, 1],
        max_num_points=30,
        max_voxels=20000,
        full_mean=False)
anchor_center = np.random.rand(100, 3)
voxels, coors, num_points_per_voxel = voxel_generator.generate(anchor_center)
print(voxels.shape, coors.shape)




```

