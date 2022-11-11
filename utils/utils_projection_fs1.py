"""
untils_projection_fs1:
14 views, SoftSilhouetteShader
for few shot of ModelNet40 & Manifold40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_shader import UntexturedSoftPhongShader
from pytorch3d.renderer import *
from typing import NamedTuple, Sequence, Union
import sys
import os

class BlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determines
            the sharpness of the edges of the shape. Higher => faces have less defined
            edges. For SplatterPhong, this is the standard deviation of the Gaussian
            kernel. Higher => splats have a stronger effect and the rendered image is
            more blurry.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1., 1., 1.)

class projection(nn.Module):
    def __init__(self, args):
        super(projection, self).__init__()
        self.lights = None
        self.args = args
    def forward(self, x, batch_size):
#        elev = torch.tensor([0., 0., 0., 0., 90., -90., 45. ,45. ,45., 45.]).repeat(batch_size)
#        elev = torch.tensor([0., 0., 0., 0., 90., -90.]).repeat(batch_size)
        elev = torch.tensor([10., 10., 10., 10.,
                             90., -90.,
                             45, 45, 45, 45,
#                             15, 15, 15, 15,
                             15, 15, 15, 15,
#                             -10, -10
                             ]).repeat(batch_size)
#        azim = torch.tensor([0., 90., 180., 270., 0., 0., 45., 135., 225., 315.]).repeat(batch_size)
#        azim = torch.tensor([0., 90., 180., 270., 0., 0.]).repeat(batch_size)
        azim = torch.tensor([0., 90., 180., 270.,
                             0., 0.,
                             45, 135, 225, 315,
#                             45, 135, 225, 315,
                             45, 135, 225, 315,
#                             90, 270
                             ]).repeat(batch_size)
#        lights = torch.tensor(
#            [[0., 0., 3.0], [0., 0., 3.0], [0., 0., 3.0], [0., 0., 3.0], [0., 0., 3.0], [0., 0., -3.0], [0., 0., 3.0], [0., 0., 3.0], [0., 0., 3.0], [0., 0., 3.0]]).repeat(
#            batch_size, 1)
#        lights = torch.tensor(
#            [[0., 0., 3.0], [3.0, 0., 0], [0., 0., -3.0], [-3.0, 0., 0], [0., 3.0, 0.], [0., -3.0, 0]]).repeat(batch_size, 1)
        lights = torch.tensor(
            [[0., 0., 3.0], [3.0, 0., 0], [0., 0., -3.0], [-3.0, 0., 0],
             [0., 3.0, 0.], [0., -3.0, 0],
             [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1],
#             [1, -1, 1], [1, -1, -1], [-1, -1, -1], [-1, -1, 1],
             [2, 0, 2], [2, 0, -2], [-2, 0, -2], [-2, 0, 2],
#             [3.0, 0., 0.], [-3.0, 0., 0.]
             ]).repeat(batch_size, 1)
        self.lights = PointLights(device=self.args.device, location=lights)
        R, T = look_at_view_transform(dist=1.7, elev=elev, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.args.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.args.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=None,
            bin_size=0
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
#            shader=UntexturedSoftPhongShader(
#                device=self.args.device,
#                blend_params=BlendParams()
#            ),
            shader=SoftSilhouetteShader(
                blend_params=BlendParams()
            )
        )
        meshes = x.extend(self.args.mesh_views)
        images = self.renderer(meshes, cameras=self.cameras)# , lights=self.lights
        images[:, ..., 3] = 1. - images[:, ..., 3]
        images[:, ..., 0] = images[:, ..., 3]
        images[:, ..., 1] = images[:, ..., 3]
        images[:, ..., 2] = images[:, ..., 3]
        return images