# SpConv: PyTorch Spatially Sparse Convolution Library

Some comments about the usage of the library. The original README is in [README_original.md](./README_original.md). Also check it!

### Installation Notes
- The CUDA path is hard-coded so it's better to first `ln` `/usr/local/cuda` to the correct version of CUDA if multiple CUDA is used. Ref: [https://github.com/traveller59/spconv/issues/78](https://github.com/traveller59/spconv/issues/78)

### Difference between [https://github.com/traveller59/spconv](https://github.com/traveller59/spconv)

- fix bug in `points_to_voxel_3d_np_mean` according to [https://github.com/traveller59/spconv/issues/135][https://github.com/traveller59/spconv/issues/135]
- set `full_mean` in `VoxelGenerator` to be `False` in default otherwise there will be assertion error.

These two do not affect actually since we should use `VoxelGeneratorV2` and set `full_mean=False`

## Usage

### Utils
- DO NOT use `VoxelGenerator` since there seem to be bugs in this function. USE `VoxelGeneratorV2` instead.

An example to check:
```
import spconv
import torch
anchor_center = np.random.rand(4, 3)
# USE THE DEFAULT VALUE SO THAT: `full_mean=False` and `block_filtering=False` (i.e. do not filter out any points)
voxel_generator = spconv.utils.VoxelGeneratorV2(voxel_size=[0.5] * 3,
                                                point_cloud_range=[0, 0, 0, 1, 1, 1],
                                                max_num_points=4,
                                                max_voxels=800000)
res = voxel_generator.generate(anchor_center)
for k, v in res.items():
    print('-' * 30)
    print(k)
    print(v)

---------------------------------------------------------

In []: anchor_center
Out[]: 
array([[0.6637, 0.0214, 0.7978],
       [0.5229, 0.3244, 0.5621],
       [0.3089, 0.668 , 0.805 ],
       [0.5843, 0.5398, 0.7831]])

Output>>
------------------------------
voxels # input anchor_center is of shape (4, 3), so voxels will be (4, max_num_points, 3)
[[[0.6637 0.0214 0.7978]
  [0.5229 0.3244 0.5621]
  [0.     0.     0.    ]
  [0.     0.     0.    ]]

 [[0.3089 0.668  0.805 ]
  [0.     0.     0.    ]
  [0.     0.     0.    ]
  [0.     0.     0.    ]]

 [[0.5843 0.5398 0.7831]
  [0.     0.     0.    ]
  [0.     0.     0.    ]
  [0.     0.     0.    ]]]
------------------------------
coordinates # with the same shape[0] as voxels, int32 tensor. NOTE: **zyx** format!!!!
[[1 0 1]
 [1 1 0]
 [1 1 1]]
------------------------------
num_points_per_voxel
[2 1 1]
------------------------------
voxel_point_mask
[[[1.]
  [1.]
  [0.]
  [0.]]

 [[1.]
  [0.]
  [0.]
  [0.]]

 [[1.]
  [0.]
  [0.]
  [0.]]]
------------------------------
voxel_num
3



```

### Convolution

- `SparseConv2d` is `SC` in the original paper `3D Semantic Segmentation with Submanifold Sparse Convolutional Networks`
- `SubMConv2d` is `SSC` in the original paper `3D Semantic Segmentation with Submanifold Sparse Convolutional Networks`
- `SparseConvTranspose2d` is the inverse operation of `SparseConv2d`. It is suggested not to use it since it will generate too much new points according to [https://github.com/traveller59/spconv/issues/149#issuecomment-639351324](https://github.com/traveller59/spconv/issues/149#issuecomment-639351324)
- `SparseInverseConv2d`: use this for deconv! [https://github.com/traveller59/spconv/issues/18#issuecomment-463537246](https://github.com/traveller59/spconv/issues/18#issuecomment-463537246)

### Understanding
- For sparse convolution operation: [https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1](https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1)
- `SparseInverseConv2d`: assume `SparseInverseConv2d` has the same `indice_key` as the previous `SparseConv2d` or `SubMConv2d` (both are ok). `indice_key` actually refers to some variables defining how each output pixel is related to the input kernel & input pixel. 
e.g. in `SparseConv2d` or `SubMConv2d`, output `y[i]` is the weight average of input `x[j], j \in N(i)`, here `N(i)={j_1, j_2, ..., j_k}` such that `y[i] = \sum_{j \in N(i)} w_j * x[j]`. 
Here `x[j]` themselves define the *active set* `A`, i.e. `x[j]` is active is equivalent to `j \in A`.
Then in `SparseInverseConv2d`, output `x[j]` is the weight average of input `{y[i] | N(i) contains j}`. And this kind of weight average is computed only in the same active set position `A`, i.e. we only compute output in position `j \in A`. Therefore, the output has the same active set as the raw original input (i.e. recover the original sparse set)

The understanding of deconv (3x3, stride=2) example is in: [https://www.zhihu.com/question/43609045/answer/145192432](https://www.zhihu.com/question/43609045/answer/145192432)

## Original examples

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


## FINAL MY EXAMPLE (basic tutorial of spconv)
```
import spconv
import numpy as np
np.random.seed(123)

import torch
import torch.nn as nn

####### points to voxels
anchor_center = np.random.rand(20, 3)
voxel_size = 0.2
voxel_generator = spconv.utils.VoxelGeneratorV2(voxel_size=[voxel_size] * 3,
                                                point_cloud_range=[0, 0, 0, 1, 1, 1],
                                                max_num_points=20,
                                                max_voxels=800000)
res = voxel_generator.generate(anchor_center)


####### voxel feature extractor: MeanVFE
voxel_features = torch.sum(torch.FloatTensor(res['voxels']), dim=1)
normalizer = torch.clamp_min(torch.FloatTensor(res['num_points_per_voxel']).view(-1, 1), min=1.0)
features = voxel_features / normalizer

####### create sparse tensor
# class SparseConvTensor(object):
#     def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
#         """
#         Args:
#             features: [num_points, num_features] feature tensor
#             indices: [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
#             spatial_shape: spatial shape of your sparse data
#             batch_size: batch size of your sparse data
#             grid: pre-allocated grid tensor. should be used when the volume of spatial shape is very large.
#         """

# remember to put to gpu: tocuda(): https://github.com/traveller59/spconv/issues/18#issuecomment-479809548
indices = torch.cat([torch.zeros((res['coordinates'].shape[0], 1)).int(), torch.IntTensor(res['coordinates'])], axis=1) 
spatial_shape = [int(1 / voxel_size)] * 3
batch_size = 1
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

####### to dense
# convert sparse tensor to dense NCHW tensor. The shape is: (B, C, Z, Y, X), 
# B is batch size, C is feature channel, the order is zyx. We can directly use `indices` to access `features`
# `x.dense()[indices[i, 0], :, indices[i, 1], indices[i, 2], indices[i, 3]]` is exactly `features[i]`
# Note that `indices.shape[0] == features.shape[0]`

x_dense = x.dense() 

print(x.sparity) # helper function to check sparity. 

####### build sparse network
# It is a common usage to set all bias=False in all sparse conv/deconv 
#
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 3, 3, bias=False, indice_key="subm0"),
            spconv.SparseConv3d(3, 3, 3, 2, bias=False, indice_key="spconv0"),
            spconv.SparseInverseConv3d(3, 1, 3, bias=False, indice_key="spconv0"), # need provide kernel size to create weight
        )

    def forward(self, x):
        return_list = []
        for layer in self.net:
            x = layer(x)
            return_list.append(x)
        return return_list

net = Model()
return_list = net(x)


####### visualize the intermediate layers for understanding the active sets
#
num_layers = len(return_list)
spatial_size = int(1 / voxel_size)
import matplotlib.pyplot as plt
def rescale_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
# for i in range(spatial_size):
for i in range(2):
    print(i)
    plt.subplot(num_layers + 1, spatial_size, i + 1)
    ximg = np.transpose(x.dense()[0][:, i, :, :].numpy(), [1, 2, 0])
    plt.imshow(rescale_img(ximg))
    plt.title('ximg')

    for lid in range(num_layers):
        plt.subplot(num_layers + 1, spatial_size, i + 1 + spatial_size * (lid+1))
        yimg = np.transpose(return_list[lid].dense()[0][:, i, :, :].detach().numpy(), [1, 2, 0])
        plt.imshow(rescale_img(yimg))
        plt.title(f'{lid}')
        print(lid, yimg.shape)

plt.show()


```
