import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage.morphology import binary_erosion
from utils.general import get_dataset_path, json_load
 

def mix(fg_img, mask_fg, bg_img, do_smoothing, do_erosion):
    """ Mix fg and bg image. Keep the fg where mask_fg is True. """
    assert bg_img.shape == fg_img.shape
    fg_img = fg_img.copy()
    mask_fg = mask_fg.copy()
    bg_img = bg_img.copy()

    if len(mask_fg.shape) == 2:
        mask_fg = np.expand_dims(mask_fg, -1)

    if do_erosion:
        mask_fg = binary_erosion(mask_fg, structure=np.ones((5, 5, 1)) )

    mask_fg = mask_fg.astype(np.float32)

    if do_smoothing:
        mask_fg = gaussian_filter(mask_fg, sigma=0.5)

    merged = (mask_fg * fg_img + (1.0 - mask_fg) * bg_img).astype(np.uint8)
    return merged


class DatasetUnsupervisedMultiview(Dataset):
    def __init__(self,  root=None, transform=None, cross_camera=False,
                 cross_time=False, cross_bg=False):
        print("Starting to load multiview data.")
        if root is None:
            self.base_path = get_dataset_path()
        else:
            self.base_path = root
        self.cross_camera = cross_camera
        self.cross_time = cross_time
        self.cross_bg = cross_bg

        self.subsets = ['gs', 'merged', 'homo', 'color_auto']  # 'color_sample']
        #self.subsets = ['gs', ]

        if self.cross_bg:
            self.subsets = ['mask_hand']

        self.camsets = {
            # neighboring  # opposing
            0: [1, 4, 7, 0],  #  [3]
            1: [0, 2, 6, 1],  #  [5]
            2: [1, 3, 4, 2],  #  [7]
            3: [2, 5, 6, 3],  #  [0]
            4: [0, 2, 5, 4],  #  [6]
            5: [3, 4, 7, 5],  #  [1]
            6: [1, 3, 7, 6],  #  [4]
            7: [0, 5, 6, 7],  #  [2]
        }  # for each cam which cams are considered good partners

        self.timeset = (-1, 0, 1)
        # load meta info file
        self.meta_info = json_load(os.path.join(self.base_path, 'meta.json'))
        self.dataset = json_load(os.path.join(self.base_path, 'index_mv_unsup_weak.json'))

        random.shuffle(self.dataset)
        self.size = len(self.dataset)

        print("Using dataset: ", self.base_path)
        print("cross_camera", cross_camera, "size", len(self.camsets[0]))
        print("cross_time", cross_time, "size", len(self.timeset))
        print("cross_bg", cross_bg)
        print('Sampling from subsets', self.subsets)
        print('Sampling from %d time steps' % self.size)

        assert transform is not None
        #assert not isinstance(transform, moco_loader.TwoCropsTransform)
        self.transform = transform


    def __len__(self):
        return self.size * 8

    def __getitem__(self, idx):
        sid, fid, K_list, M_list = self.dataset[idx % self.size]
        # roll for a random camera
        cid1 = random.randint(0, 7)

        if self.cross_camera:
            cid2 = random.choice(self.camsets[cid1])
        else:
            cid2 = cid1

        fid1 = fid
        if self.cross_time:
            s_max = len(self.meta_info['is_train'][sid])-1
            fid2 = min(max(0, fid + random.choice(self.timeset)), s_max)
        else:
            fid2 = fid

        if self.meta_info['is_train'][sid][fid]:
            subset1 = random.choice(self.subsets)
            subset2 = random.choice(self.subsets)
        else:
            subset1 = 'test'
            subset2 = 'test'

        try:
            # read the frame
            sample1 = self.read(sid, fid1, cid1, subset1)
            sample2 = self.read(sid, fid2, cid2, subset2)

            if self.transform is not None:
                sample1 = self.transform(sample1)
                sample2 = self.transform(sample2)
            return (sample1, sample2), 0
        except FileNotFoundError as e:
            # print(e)
            return self.__getitem__(idx)


    def read(self, sid, fid, cid, subset):
        if subset == 'mask_hand':
            return self.read_rnd_background(sid, fid, cid, subset)

        if subset == 'gs' or subset == 'test':
            img_path = 'rgb/%04d/cam%d/%08d.jpg' % (sid, cid, fid)
        else:
            img_path = 'rgb_%s/%04d/cam%d/%08d.jpg' % (subset, sid, cid, fid)

        # read samples
        path = os.path.join(self.base_path, img_path)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def read_rnd_background(self, sid, fid, cid, subset):
        # sample rnd background
        base_path = '/misc/lmbraid18/zimmermc/'
        rid = random.randint(0, 1230)
        bg_image_new_path = os.path.join(base_path, 'background_subtraction/background_examples/bg_new/%05d.jpg' % rid)
        bg_img_new = Image.open(bg_image_new_path)

        mask_path = 'mask_hand/%04d/cam%d/%08d.jpg' % (sid, cid, fid)
        mask_path = os.path.join(self.base_path, mask_path)
        mask_fg = Image.open(mask_path)

        img_path = 'rgb/%04d/cam%d/%08d.jpg' % (sid, cid, fid)
        img_path = os.path.join(self.base_path, img_path)
        fg_img = Image.open(img_path)


        bg_img_new = np.asarray(bg_img_new.resize(fg_img.size))
        fg_img = np.asarray(fg_img)
        mask_fg = (np.asarray(mask_fg) / 255.)[:, :, None]

        merged = mix(fg_img, mask_fg, bg_img_new, do_smoothing=True, do_erosion=True)

        return Image.fromarray(merged)


def get_dataset(batch_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1.0, 1.0, 1.0])

    img_size = 112  # running with 224 resolution did not improve results
    print("Warning: Un-comment augmentations for training")

    # these are the agumentations as we use for our moco pre-training
    # please un-comment the gaussian blue and normalization before training
    augmentation = [
        transforms.RandomAffine(10),
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize
    ]

    dataset = DatasetUnsupervisedMultiview(None, transforms.Compose(augmentation),
                                           cross_camera=False,
                                           cross_time=False,
                                           cross_bg=False)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8)


if __name__ == '__main__':
    batch_size = 3
    d = get_dataset(batch_size)
    
    for sample in d:
        data, label = sample
        for i in range(batch_size):
            img = data[0][i].numpy().transpose(1, 2, 0)
            img_aug = data[1][i].numpy().transpose(1, 2, 0)

            fig, ax = plt.subplots(1,2)
            ax[0].imshow(img)
            ax[1].imshow(img_aug)
            plt.show()
