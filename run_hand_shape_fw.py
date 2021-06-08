""" Run forward pass on trained hand shape estimation network. """
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import argparse
import torch
import torch.nn as nn
import glob
import numpy as np
import cv2
import time, os, json
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_util import draw_hand
from utils.rendering import render_verts_faces
from manopth.manolayer import ManoLayer
from nets.ResNet import resnet50


from utils.img_util import downsample
from utils.mano_utils import apply_scaling, pred_to_mano, project
from utils.general import load_ckpt, json_load


class ManoPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False, head_type='mano')
        self.mano = ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=9)

    def forward(self, image_np, K_np, device='cpu'):
        assert image_np.shape == (224, 224, 3), 'Image shape mismatch.'
        img = np.transpose(image_np[:, :, ::-1], [2, 0, 1]).astype(np.float32) / 255.0 - 0.5
        img = np.expand_dims(img, 0)
        theta_p = self.model(
                torch.Tensor(img).to(device)
            )
        theta_p = apply_scaling(theta_p)
        poses, shapes, global_t = pred_to_mano(theta_p,
                                               torch.Tensor(K_np[None]).to(device)
                                               )
        verts_p, xyz_p = self.mano(poses, shapes, global_t)
        uv_p = project(xyz_p, torch.Tensor(K_np[None]).to(device))

        verts_p_np = verts_p.detach().cpu().numpy()[0]
        xyz_p_np = xyz_p.detach().cpu().numpy()[0]
        uv_p_np = uv_p.detach().cpu().numpy()[0]

        img_shape = np.array([[image_np.shape[0], image_np.shape[1]]])
        mask_p, _  = render_verts_faces(verts_p,
                                      self.mano.th_faces[None],
                                      K_np[None], np.eye(4)[None], img_shape)
        mask_np = mask_p[0].detach().cpu().numpy()[0].transpose([1, 2, 0])
        return verts_p_np, xyz_p_np, uv_p_np, mask_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hanco_path', type=str, help='Path to where HanCo dataset is stored.')
    parser.add_argument('--sid', type=int, help='Sequence ID.', default=110)
    parser.add_argument('--cid', type=int, help='Camera ID.', default=3)
    parser.add_argument('--fid', type=int, help='Frame ID.', default=0)
    args = parser.parse_args()

    assert os.path.exists(args.hanco_path), 'Path to HanCo not found.'
    assert os.path.isdir(args.hanco_path), 'Path to HanCo doesnt seem to be a directory.'


    img_path = os.path.join(args.hanco_path, f'rgb/{args.sid:04d}/cam{args.cid}/{args.fid:08d}.jpg')
    calib_path = os.path.join(args.hanco_path, f'calib/{args.sid:04d}/{args.fid:08d}.json')

    assert os.path.exists(img_path), f'Image not found: {img_path}'
    assert os.path.exists(calib_path), f'Calibration not found: {calib_path}'

    img = cv2.imread(img_path)
    K = np.array(json_load(calib_path)['K'][3])

    # Load network
    model = ManoPredictor()
    state_dict = torch.load('ckpt/model_mano.pth')
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # forward pass
    with torch.no_grad():
        verts_xyz_p, joints_xyz_p, joints_uv_p, mask_p = model.forward(img, K, 'cuda')

        # vis rgb image with predicted skeleton
        img_vis = draw_hand(img.copy(), joints_uv_p, kp_style=(2, 1), order='uv', img_order='bgr')

        # vis rendered mask with predicted skeleton
        mask_p = np.clip(mask_p*255, 0, 255).astype(np.uint8)
        mask_vis = draw_hand(mask_p.copy(), joints_uv_p, kp_style=(2, 1), order='uv', img_order='bgr')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_vis[:, :, ::-1]), ax[0].set_title('rgb+pred skel')
    ax[1].imshow(mask_vis[:, :, ::-1]), ax[1].set_title('pred shape+skel')
    plt.show()

if __name__ == '__main__':
    main()
