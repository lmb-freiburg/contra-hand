import torch
import numpy as np

np.cat = np.concatenate
torch.transpose = lambda x, y: x.permute(y)


def apply_scaling(theta):
    poses, shapes, cams = slice_theta(theta)

    poses_scaled = 1.0 * poses
    shapes_scaled = 0.5 * shapes
    root = cams[:, :2]  # estimated root im image coords
    scale = cams[:, -1:]  # estimated shape scale

    root = 14.0 * root + 112.0
    scale = 125.0 * scale + 730.0
    cams_scaled = torch.cat([root, scale], -1)

    theta_scaled = torch.cat([poses_scaled, shapes_scaled, cams_scaled], -1)
    return theta_scaled


def slice_theta(theta):
    """ Slice vector of all hand shape parameters into sematically meaningful parts.
    """
    return theta[:, :48], theta[:, 48:58], theta[:, -3:]


def slice_cams(cams):
    """
    Returns translation in uv and scale.
    """
    return cams[:, :2], cams[:, -1:]


def project(xyz, K, fw=torch):
    """ Project points into the camera. """
    uv = fw.matmul(xyz, fw.transpose(K, [0, 2, 1]))
    uv = uv[:, :, :2] / uv[:, :, -1:]
    return uv


def unproject(points2d, K, z=None, K_is_inv=False, fw=torch):
    """ Unproject a 2D point of camera K to distance z.
    """
    batch = K.shape[0]
    points2d = fw.reshape(points2d, [batch, -1, 2])
    points2d_h = fw.cat([points2d, fw.ones_like(points2d[:, :, :1])], -1)  # homogeneous

    if K_is_inv:
        K_inv = K
    else:
        if fw == torch:
            K_inv = fw.inverse(K)
        else:
            K_inv = fw.linalg.inv(K)

    points3D = fw.matmul(points2d_h, fw.transpose(K_inv, [0, 2, 1]))  # 3d point corresponding to the estimate image point where the root should go to
    if z is not None:
        z = fw.reshape(z, [batch, -1, 1])
        points3D = points3D * z
    return points3D


def trafoPoints(xyz, M, fw=torch):
    """ Transforms points into another coordinate frame. """
    xyz_h = fw.cat([xyz, fw.ones_like(xyz[:, :, :1])], 2)
    xyz_cam = fw.matmul(xyz_h, fw.transpose(M, [0, 2, 1]))
    xyz_cam = xyz_cam[:, :, :3] / xyz_cam[:, :, -1:]
    return xyz_cam


def calc_global_translation(trans_uv, scale, K, fw=torch):
    """ Calculate global translation from uv position and scale.
    """
    scale = fw.reshape(scale, [-1, 1, 1])
    z = 0.5 * (K[:, :1, :1] + K[:, 1:2, 1:2]) / scale  # calculate root depth from scale

    # calculate and apply global translation
    global_t = unproject(trans_uv, K, z, fw=fw)  # unprojection of the estimated mano root using the estimated depth
    return global_t, z


def calc_global_translation_from_theta(theta, K, fw=torch):
    """ Calculate global translation from uv position and scale.
    """
    _, _, cams = slice_theta(theta)
    trans_uv, scale = slice_cams(cams)
    return calc_global_translation(trans_uv, scale, K, fw=fw)


def pred_to_mano(theta, K, fw=torch):
    """ Convert predicted theta into MANO parameters.
    """
    poses, shapes, cams = slice_theta(theta)
    trans_uv, scale = slice_cams(cams)
    global_t, _ = calc_global_translation(trans_uv, scale, K, fw=fw)
    return poses, shapes, global_t


def mano_to_vector(poses, shapes, global_t, K, fw=torch):
    """ Given the semantic parts of the mano shape model, create a parameter vector out of it (which will be estimated by networks)

        poses and global_t must already be in the cameras 3D coordinate frame.
    """
    # project 3D point into cam
    trans_uv = project(global_t, K, fw=fw)

    # find scale = focal_length / depth
    scale = 0.5*(K[:, 0, 0] + K[:, 1, 1])[:, None] / global_t[:, :, -1]

    # assemble cams
    cams = fw.cat([trans_uv[:, 0], scale], -1)

    # assemble theta
    theta = fw.cat([poses, shapes, cams], -1)
    return theta
