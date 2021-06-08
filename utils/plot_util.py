from __future__ import print_function, unicode_literals
import numpy as np
import cv2


def draw_hand(image, coords_hw, vis=None, color_fixed=None, linewidth=3, order='hw', img_order='rgb',
              draw_kp=True, kp_style=None):
    """ Inpaints a hand stick figure into a matplotlib figure. """
    if kp_style is None:
        kp_style = (5, 3)

    image = np.squeeze(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)
    s = image.shape
    assert len(s) == 3, "This only works for single images."

    convert_to_uint8 = False
    if s[2] == 1:
        # grayscale case
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
        image = np.tile(image, [1, 1, 3])
        pass
    elif s[2] == 3:
        # RGB case
        if image.dtype == np.uint8:
            convert_to_uint8 = True
            image = image.astype('float32') / 255.0
        elif image.dtype == np.float32:
            # convert to gray image
            image = np.mean(image, axis=2)
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
            image = np.expand_dims(image, 2)
            image = np.tile(image, [1, 1, 3])
    else:
        assert 0, "Unknown image dimensions."

    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    if img_order == 'rgb':
        colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    color_map = {'k': np.array([0.0, 0.0, 0.0]),
                 'w': np.array([1.0, 1.0, 1.0]),
                 'b': np.array([0.0, 0.0, 1.0]),
                 'g': np.array([0.0, 1.0, 0.0]),
                 'r': np.array([1.0, 0.0, 0.0]),
                 'm': np.array([1.0, 1.0, 0.0]),
                 'c': np.array([0.0, 1.0, 1.0])}

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :].astype(np.int32)
        coord2 = coords_hw[connection[1], :].astype(np.int32)

        if (coord1[0] < 1) or (coord1[0] >= s[0]) or (coord1[1] < 1) or (coord1[1] >= s[1]):
            continue
        if (coord2[0] < 1) or (coord2[0] >= s[0]) or (coord2[1] < 1) or (coord2[1] >= s[1]):
            continue

        if color_fixed is None:
            cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), color, thickness=linewidth)
        else:
            c = color_map.get(color_fixed, np.array([1.0, 1.0, 1.0]))
            cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), c, thickness=linewidth)

    if draw_kp:
        coords_hw = coords_hw.astype(np.int32)
        for i in range(21):
            if vis[i]:
                # cv2.circle(img, center, radius, color, thickness)
                image = cv2.circle(image, (coords_hw[i, 1], coords_hw[i, 0]),
                                   radius=kp_style[0], color=colors[i, :], thickness=kp_style[1])

    if convert_to_uint8:
        image = (image * 255).astype('uint8')

    return image

