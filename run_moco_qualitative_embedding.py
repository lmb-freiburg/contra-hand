""" Script to produce the data for Figure 3 of the paper. """
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from run_moco_fw import ModelWrap


m = ModelWrap()
data_path = m.base_path

cossim = lambda x, y: np.sum(x*y)/np.linalg.norm(x, 2)/np.linalg.norm(y, 2)


def show(path1, path2, save_to=None):
    print("show('%s', '%s')" % (path1, path2))

    s = cossim(m.run(path1), m.run(path2))

    img1 = cv2.imread(os.path.join(data_path, path1))
    img2 = cv2.imread(os.path.join(data_path, path2))
    if save_to is not None:
        cv2.imwrite(save_to + '_0.png', img1)
        cv2.imwrite(save_to + '_1.png', img2)
        with open(save_to + '_s.txt', 'w') as fo:
            fo.write('%f' % s)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1[:, :, ::-1])
    ax[1].imshow(img2[:, :, ::-1])
    ax[1].set_title('score = %.3f' % s)
    plt.show()


# 1. show same image image pairs with different backgrounds is encoded the same
for i in (0, 5, 7):
    show('0108/cam4/00000017_%d.jpg' % i, '0108/cam4/00000017_%d.jpg' % (i+1),
         save_to='./moco_vis_ex/same_sample_diff_bg/%02d' % i)

# 2. Show similar poses are encoded similarly
i = 2
show('0007/cam4/00000016_5.jpg', '0007/cam4/00000017_4.jpg',
         save_to='./moco_vis_ex/similar_poses/%02d' % i)
i = 3
show('0011/cam4/00000004_0.jpg', '0011/cam4/00000005_1.jpg',
         save_to='./moco_vis_ex/similar_poses/%02d' % i)
i = 4
show('0011/cam4/00000011_1.jpg', '0011/cam4/00000012_2.jpg',
         save_to='./moco_vis_ex/similar_poses/%02d' % i)


# 3. Different views are encoded similarly
i = 0
show('0108/cam4/00000026_3.jpg', '0108/cam3/00000026_5.jpg',
   save_to='./moco_vis_ex/diff_view/%02d' % i)
i = 1
show('0011/cam4/00000012_2.jpg', '0011/cam3/00000012_5.jpg',
   save_to='./moco_vis_ex/diff_view/%02d' % i)
i = 3
show('0011/cam4/00000005_1.jpg', '0011/cam3/00000005_6.jpg',
   save_to='./moco_vis_ex/diff_view/%02d' % i)


# 3. Different poses are encoded differently
i = 0
show('0007/cam4/00000000_3.jpg', '0007/cam4/00000018_3.jpg',
         save_to='./moco_vis_ex/diff_poses/%02d' % i)
i = 1
show('0108/cam4/00000007_0.jpg', '0108/cam4/00000019_0.jpg',
         save_to='./moco_vis_ex/diff_poses/%02d' % i)
i = 3
show('0007/cam3/00000015_2.jpg', '0108/cam3/00000026_2.jpg',
         save_to='./moco_vis_ex/diff_poses/%02d' % i)
