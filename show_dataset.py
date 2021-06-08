""" Iterate HanCo dataset and show how to work with data. """
import os, argparse, json
import numpy as np
import cv2
import matplotlib.pyplot as plt


from utils.plot_util import draw_hand

def example_meta_data(args):
    meta_file = os.path.join(args.hanco_path, 'meta.json')
    with open(meta_file, 'r') as fi: 
        meta_data = json.load(fi)
    print(type(meta_data))  # Its a dict
    print(meta_data.keys())  # Its keys are: 'is_train', 'subject_id', 'is_valid', 'object_id', 'has_fit'

    for k, v in meta_data.items():
        print(k, type(v), len(v), v[0][:3], v[-1][:3])  # these are all lists of length 1518 (= one entry for each sequence), each entry is another list representing the frames of the sequence

    # is_train: bool, True if recorded with green screen background
    # subject_id: int, Unique identifier for the human performer
    # is_valid: bool, True if there is a validated MANO shape fit
    # object_id: int, Unique identifier for the object used. None for sequences w/o object interaction
    # has_fit: bool, True if there is a MANO shape fit. Potentially, not validated


def example_show_data(args, sid):
    """
        sid: Sequence id: int, in [0, 1517]
    """
    meta_file = os.path.join(args.hanco_path, 'meta.json')
    with open(meta_file, 'r') as fi: 
        meta_data = json.load(fi) 
            
    print(f"\nShowing sequence {sid} with {len(meta_data['is_train'][sid])} frames.")
    
    # iterate frames of this sequence
    for fid in range(len(meta_data['is_train'])):
        print(f"fid={fid},\n"
              f"is_train={meta_data['is_train'][sid][fid]},\n"
              f"subject_id={meta_data['subject_id'][sid][fid]},\n"
              f"is_valid={meta_data['is_valid'][sid][fid]},\n"
              f"object_id={meta_data['object_id'][sid][fid]},\n"
              f"has_fit={meta_data['has_fit'][sid][fid]}")
        rgb_list = list()
        for cid in range(8): # iterate cameras
            rgb_path = os.path.join(args.hanco_path, f'rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg')
            rgb_list.append(
                cv2.imread(rgb_path)[:, :, ::-1]
            )
    
        # show
        fig, ax = plt.subplots(1, 8)
        for j, img in enumerate(rgb_list):
            ax[j].imshow(img)
            ax[j].set_xticks([], [])
            ax[j].set_yticks([], [])
        plt.show()

        if fid > 3:
            # we deliberately stop showing after some samples
            break

def example_show_keypoints(args, sid, fid, cid):
    # load image
    image_file = os.path.join(args.hanco_path, f'rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg')
    img = cv2.imread(image_file)[:, :, ::-1]

    # load keypoints
    kp_data_file = os.path.join(args.hanco_path, f'xyz/{sid:04d}/{fid:08d}.json')
    with open(kp_data_file, 'r') as fi:
        kp_xyz = np.array(json.load(fi))
    print('kp_xyz', kp_xyz.shape, kp_xyz.dtype)  # 21x3, np.float64, world coordinates

    # load calibration
    calib_file = os.path.join(args.hanco_path, f'calib/{sid:04d}/{fid:08d}.json')
    with open(calib_file, 'r') as fi:
        calib = json.load(fi)

    # project points
    M_w2cam = np.array(calib['M'])[cid]
    K = np.array(calib['K'])[cid]
    kp_xyz_cam = np.matmul(kp_xyz, M_w2cam[:3, :3].T) + M_w2cam[:3, 3][None]  # in camera coordinates
    kp_xyz_cam = kp_xyz_cam / kp_xyz_cam[:, -1:]
    kp_uv = np.matmul(kp_xyz_cam, K.T)
    kp_uv = kp_uv[:, :2] / kp_uv[:, -1:]

    # show
    img = draw_hand(img, kp_uv, order='uv', img_order='rgb')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


def example_show_shape(args, sid, fid, cid):
    import torch
    from manopth.manolayer import ManoLayer
    from utils.mano_utils import pred_to_mano, project, trafoPoints
    from utils.rendering import render_verts_faces

    # load image
    image_file = os.path.join(args.hanco_path, f'rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg')
    img = cv2.imread(image_file)[:, :, ::-1]

    # load calibration
    calib_file = os.path.join(args.hanco_path, f'calib/{sid:04d}/{fid:08d}.json')
    with open(calib_file, 'r') as fi: 
        calib = json.load(fi)

    # load shape in world space
    kp_data_file = os.path.join(args.hanco_path, f'shape/{sid:04d}/{fid:08d}.json')
    with open(kp_data_file, 'r') as fi: 
        mano_w = json.load(fi)
    for k, v in mano_w.items():
        print(k, np.array(v).shape) # a dict of pose, shape and global_t

    # load shape in camera space
    kp_data_file = os.path.join(args.hanco_path, f'shape/{sid:04d}/cam{cid}/{fid:08d}.json')
    with open(kp_data_file, 'r') as fi: 
        mano_cam = np.array(json.load(fi))[None]
    print('mano_vec', mano_cam.shape) # parameter vector
    pose_cam, shape_cam, global_t_cam = pred_to_mano(mano_cam, np.array(calib['K'])[cid][None], fw=np)

    # render shape masks
    def render_hand(poses, shapes, global_t, img_shape, K, M=None, center_idx=None):
        if M is None:
            M = np.eye(4)

        mano = ManoLayer(use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=center_idx)

        verts, xyz = mano(poses, shapes, global_t)
        uv = project(trafoPoints(xyz, torch.Tensor(M)[None]), torch.Tensor(K)[None])
        mask, _  = render_verts_faces(verts,
                                      mano.th_faces[None],
                                      K[None], M[None], img_shape[None], device='cpu')


        mask = mask[0].detach().cpu().numpy()[0]
        uv = uv.detach().cpu().numpy()[0]
        return mask, uv

    mask1, uv1 = render_hand(torch.Tensor(mano_w['poses']), 
                             torch.Tensor(mano_w['shapes']), 
                             torch.Tensor(mano_w['global_t']),
                             np.array(img.shape[:2]),
                             np.array(calib['K'][cid]),
                             np.array(calib['M'][cid]))

    mask2, uv2 = render_hand(torch.Tensor(pose_cam), 
                             torch.Tensor(shape_cam), 
                             torch.Tensor(global_t_cam),
                             np.array(img.shape[:2]),
                             np.array(calib['K'][cid]),
                             center_idx=9)

    # show
    img1 = draw_hand(img, uv1, order='uv', img_order='rgb')
    img2 = draw_hand(img, uv2, order='uv', img_order='rgb')

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img1)
    ax1.imshow(mask1[0, :, :], alpha=0.5)
    ax2.imshow(img2)
    ax2.imshow(mask2[0, :, :], alpha=0.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hanco_path', type=str, help='Path to where HanCo dataset is stored.')
    args = parser.parse_args()

    assert os.path.exists(args.hanco_path), 'Path to HanCo not found.'
    assert os.path.isdir(args.hanco_path), 'Path to HanCo doesnt seem to be a directory.'


    # Example1: Meta data
    example_meta_data(args)

    # Example2: Read/Show all images of one sequence
    example_show_data(args, 110)

    # Example3: Show keypoints, calibration, camera projection
    example_show_keypoints(args, sid=110, fid=24, cid=3)

    # Example4: Render MANO shape, show 
    example_show_shape(args, sid=110, fid=24, cid=3)
