import torch
import torch.nn.functional as F
import numpy as np
import transforms3d as t3d
import pickle
from manopth.manolayer import ManoLayer

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardGouraudShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    BlendParams
)


def render_verts_faces(verts, faces,
                       K, M_obj2cam, img_shape,
                       verts_color=None, device='cuda',
                       segmentation=True, render_size=200):
    if verts_color is None:
        verts_color = np.array([205/255., 205/255., 205/255.], dtype=np.float32)
    verts_color = torch.Tensor(verts_color).to(device)

    # Load obj file
    verts_feat = torch.ones_like(verts) * verts_color
    verts, faces = verts.to(device), faces.to(device)
    verts_feat = verts_feat.to(device)
    tex = TexturesVertex(verts_features=verts_feat)
    mesh = Meshes(verts.to(device),
                  faces.to(device),
                  tex)

    # Convert coordinate frames: pytorch3d X left, Y up --> CV X right, Y down
    # Rotate 180deg around z axis
    M_corr = np.eye(4)
    M_corr[:3, :3] = t3d.euler.euler2mat(0.0, .0, np.pi)
    M_obj2cam = np.matmul(M_corr, M_obj2cam)

    # setup camera
    focal = np.stack([K[:, 0, 0], K[:, 1, 1]], -1)
    pp = np.stack([K[:, 0, 2], K[:, 1, 2]], -1)
    img_shape = np.stack([img_shape[:, 1], img_shape[:, 0]], -1)
    R = np.transpose(M_obj2cam[:, :3, :3], [0, 2, 1])
    T = M_obj2cam[:, :3, 3]
    cameras = PerspectiveCameras(focal_length=focal,
                                 principal_point=pp,
                                 R=R,
                                 T=T,
                                 image_size=img_shape,
                                 device=device)

    raster_settings = RasterizationSettings(
        image_size=render_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    if segmentation:
        lights = PointLights(location=((1, 1, 0), ),
                             ambient_color=((1.0, 1.0, 1.0),),
                             diffuse_color=((0.0, 0.0, 0.0),),
                             specular_color=((0.1, 0.1, 0.1),),
                             device=device)

        shader = HardGouraudShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(.0, .0, .0))
        )
    else:

        d = 0.3 # diffuse
        a = 1.0-d  # ambient
        lights = PointLights(location=((1, 1, 0),),
                             diffuse_color=((d, d, d),),
                             ambient_color=((a, a, a),),
                             specular_color=((0.1, 0.1, 0.1),),
                             device=device)

        shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=((.0, .0, .0),))
        )

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    fragments = rasterizer(mesh)

    image = renderer(mesh)
    im_out, dep_out = list(), list()
    for i, (w, h) in enumerate(img_shape):
        im_out.append(
            F.interpolate(
                image[i:i+1, :, :, :3].permute([0, 3, 1, 2]),
                (h, w))
        )
        dep_out.append(
            F.interpolate(
                fragments.zbuf[i:i+1, :, :, :1].permute([0, 3, 1, 2]),
                (h, w))
        )
    return im_out, dep_out


