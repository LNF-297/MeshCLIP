"""
UntexturedSoftPhongShader
For mesh without textual and material information
"""
import warnings
from typing import Optional

import torch
import torch.nn as nn

from pytorch3d.common.datatypes import Device
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.splatter_blend import SplatterBlender
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments

from pytorch3d.renderer.mesh.shading import (
    Tuple,
    interpolate_face_attributes,
    _apply_lighting,

)
from pytorch3d.renderer.mesh.shader import ShaderBase

def _phong_shading_with_pixels(
    meshes, fragments, lights, cameras, materials, texels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras/
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords_in_camera, pixel_normals, lights, cameras, materials
    )
    if texels is not None:
        colors = (ambient + diffuse) * texels + specular
        return colors, pixel_coords_in_camera
    else:
        colors = ambient + diffuse + specular
        return colors, pixel_coords_in_camera

def phong_shading(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _ = _phong_shading_with_pixels(
        meshes, fragments, lights, cameras, materials, texels
    )
    return colors

class UntexturedSoftPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        texels = None
        if meshes.textures is not None:
            texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar,
        )
        return images

class UntexturedHardPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
#        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
#            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images
