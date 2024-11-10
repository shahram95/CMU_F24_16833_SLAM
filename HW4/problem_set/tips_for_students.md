## Dependency

This homework is mainly depending on numpy and open3d. The latter is a cross-platform library, so Windows can run it smoothly. For Ubuntu 16.04, please install open3d==0.9.0. If you encounter any issue, please let us know.

## Vectorization

Vectorization with advanced indexing is strongly recommended in your implementation. You can find many example usages in the code skeleton, and if you have any trouble understanding how to use it, first check the documentation, then come to OH if necessary.

## Coordinate accessing

There are also a handful of image indexing. Keep in mind, images are alwasy stored in the (H, W, C) format, where H is height, W is width, C is channel. We also usually use u, v coordinates instead of x, y in the image space, and in order to access a pixel, visiting image[v, u] is a convention.

## The linear system and the solver

The linear system for ICP is (N, 6). In the handout it is phrased by QR or LU fomulation – this is basically referring to writing the system in either $Ax=b$ or $A^T Ax=A^Tb$. The solution could be straightforward, and you can simply use numpy’s dense linalg solvers. The reason why I emphasize the QR or LU formulation is that

I hope you can view the ICP problem in the observation-measurement-state formulation, which has been covered in our previous homeworks.
Implementation wise they could lead to different tricks of acceleration to make the system efficient (though not mandatory for the assignment).

## Fusion

In the point based fusion paper, an uncertainty-aware α is used as the weight for the point to be fused. In our setup this is simplified by α=1 . Also, weight average should be applied to colors. In case you don’t realize, we use self.weight to represent ‘confidence’ in the paper.

## Dataset

The tar.gz file is a compressed file, just like .zip. Please decompress it to a folder before running python code.


We hope you enjoy this homework!

