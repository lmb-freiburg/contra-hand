from __future__ import print_function, unicode_literals
import numpy as np
import cv2


def random_occlude(img, color):
    # randomly drop some spatially correlated faces
    N = np.random.randint(3, 8)
    points = np.random.rand(N, 2) * np.array([[img.shape[1], img.shape[0]]])
    points = np.round(points).astype(np.int32)
    img = cv2.fillConvexPoly(img.copy(), points, color)
    return img

def apply_colormap(class_map):
    """ Given a indexed map image this function outputs a RGB image corresponding with a jet color map. """
    image_out_r = np.zeros_like(class_map).astype('float32')
    image_out_g = np.zeros_like(class_map).astype('float32')
    image_out_b = np.zeros_like(class_map).astype('float32')
    import matplotlib.cm as cm

    # create index map with known values
    values = np.unique(class_map)

    # create colors according to color palette
    colors = cm.jet(np.linspace(0, 1, len(values)))

    for i, val in enumerate(values):
        if i == 0:
            continue
        mask = (class_map == val)
        image_out_r[mask] = colors[i, 0]
        image_out_g[mask] = colors[i, 1]
        image_out_b[mask] = colors[i, 2]
    image_out = np.stack([image_out_r, image_out_g, image_out_b], -1)
    image_out = np.round(image_out * 255).astype(np.uint8)
    return image_out


def plot_bb(ax, bb, color='r'):
    """ Draws a bounding box. """
    points = list()
    points.append([bb[0, 0], bb[0, 1]])
    points.append([bb[1, 0], bb[0, 1]])
    points.append([bb[1, 0], bb[1, 1]])
    points.append([bb[0, 0], bb[1, 1]])
    points.append([bb[0, 0], bb[0, 1]])
    points = np.array(points)
    ax.plot(points[:, 1], points[:, 0], color)


def plot_bb_ltrb(ax, pt_list, color='r', linewidth=2.0):
    """ Draws a bounding box. """
    points = list()
    points.append([pt_list[0], pt_list[1]])
    points.append([pt_list[0], pt_list[3]])
    points.append([pt_list[2], pt_list[3]])
    points.append([pt_list[2], pt_list[1]])
    points.append([pt_list[0], pt_list[1]])
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], color, linewidth=linewidth)


def draw_bb_ltrb(image, bb, color=None, text=None, linewidth=8):
    """ Inpaints a bounding box. """
    image = image.copy()
    bb = np.array(bb).copy()
    bb = np.round(bb).astype(np.int32)

    if color is None:
        color = 'r'

    color_map = {'k': np.array([0.0, 0.0, 0.0]),
                 'w': np.array([1.0, 1.0, 1.0]),
                 'b': np.array([0.0, 0.0, 1.0]),
                 'g': np.array([0.0, 1.0, 0.0]),
                 'r': np.array([1.0, 0.0, 0.0]),
                 'm': np.array([1.0, 1.0, 0.0]),
                 'c': np.array([0.0, 1.0, 1.0])}

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

    points = list()
    points.append([bb[0], bb[1]])
    points.append([bb[0], bb[3]])
    points.append([bb[2], bb[3]])
    points.append([bb[2], bb[1]])
    points.append([bb[0], bb[1]])
    points = np.array(points)
    for i in range(len(points)-1):
        c = color_map.get(color, np.array([1.0, 0.0, 0.0]))
        cv2.line(image,
                 tuple(points[i]),
                 tuple(points[i+1]),
                 c, thickness=linewidth)

    if text is not None:
        c = color_map.get(color, np.array([1.0, 0.0, 0.0]))
        cv2.putText(image, text, (bb[0], bb[1]), cv2.FONT_HERSHEY_PLAIN, 2, c, thickness=2)

    if convert_to_uint8:
        image = (image * 255).astype('uint8')
    return image


def draw_bb(image, bb, color=None, linewidth=8):
    """ Inpaints a bounding box. """
    image = image.copy()
    bb = bb.copy()
    bb = np.round(bb).astype(np.int32)

    if color is None:
        color = 'r'

    color_map = {'k': np.array([0.0, 0.0, 0.0]),
                 'w': np.array([1.0, 1.0, 1.0]),
                 'b': np.array([0.0, 0.0, 1.0]),
                 'g': np.array([0.0, 1.0, 0.0]),
                 'r': np.array([1.0, 0.0, 0.0]),
                 'm': np.array([1.0, 1.0, 0.0]),
                 'c': np.array([0.0, 1.0, 1.0])}

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

    points = list()
    points.append([bb[0, 1], bb[0, 0]])
    points.append([bb[1, 1], bb[0, 0]])
    points.append([bb[1, 1], bb[1, 0]])
    points.append([bb[0, 1], bb[1, 0]])
    points.append([bb[0, 1], bb[0, 0]])
    for i in range(len(points)-1):
        c = color_map.get(color, np.array([1.0, 0.0, 0.0]))
        cv2.line(image,
                 tuple(points[i]),
                 tuple(points[i+1]),
                 c, thickness=linewidth)

    if convert_to_uint8:
        image = (image * 255).astype('uint8')
    return image


def get_hand_color_code(rgb=True):
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
    if rgb:
        colors = colors[:, ::-1]
    return (colors*255).astype(np.uint8)


def plot_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
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

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])


def _opencv_colormap(char):
    if char is None:
        color = [255, 255, 255]
    elif char == 'r':
        color = [255, 0, 0]
    elif char == 'g':
        color = [0, 255, 0]
    elif char == 'b':
        color = [0, 0, 255]
    elif char == 'w':
        color = [255, 255, 255]
    elif char == 'k':
        color = [0, 0, 0]
    else:
        color = [0, 0, 0]
    return color


def draw_confidences(img, confidences, offset=None, font_scale=1.0, font_color=None, thickness=2, filled_color=None):
    """ Inpaints the confidences as text into the image. """
    font_color = _opencv_colormap(font_color)

    if offset is None:
        offset = np.array([35, 0])

    confidences = confidences.squeeze()

    l = np.array(img.shape[:2]).astype(np.float32) * np.array([0.05, 0.75])

    if filled_color is not None:
        filled_color = _opencv_colormap(filled_color)
        cv2.rectangle(img, (int(img.shape[1]*0.70), int(0)),
                      (int(img.shape[1]), int(img.shape[0])), color=filled_color, thickness=-1)

    for i in range(confidences.shape[0]):
        cv2.putText(img, 'kp%d=%.2f' % (i, confidences[i]),
                    (int(l[1]), int(l[0])),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, thickness, cv2.LINE_AA)
        l += offset

    cv2.putText(img, 'sum=%.2f' % (np.sum(confidences)),
                (int(l[1]), int(l[0])),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, thickness, cv2.LINE_AA)

    return img


def draw_flow_corresp(pose_img_ref, image_to, flow,
                      min_flow_mag=5.0, max_samples=50):
    """ Draws correspondences induced by the flow.
        Shows reference frame as a mask and the other image normally.
        Flow is visualized by a green dot in the reference frame and a blue line going to its location in th other frame.
    """
    pose_mask = (pose_img_ref.mean(-1) > 0.1).astype(np.uint8)*255
    merged = blend_rgb_images(image_to,
                              np.stack([pose_mask, pose_mask, pose_mask], -1),
                              0.75)

    flow_mag = np.sqrt(np.sum(np.square(flow), -1))
    flow_mask = flow_mag > min_flow_mag
    H, W = np.where(flow_mask)
    U, V = flow[:, :, 0][flow_mask], flow[:, :, 1][flow_mask]

    if U.shape[0] > max_samples:
        inds = np.random.choice(U.shape[0], max_samples, replace=False)
        U, V = U[inds], V[inds]
        H, W = H[inds], W[inds]

    for v, u, fu, fv in zip(H, W, U, V):
        merged = cv2.line(merged,
                          (u, v),
                          (int(u+fu), int(v+fv)), color=(0, 0, 255), thickness=1)

        merged = cv2.circle(merged, (u, v), radius=2, color=(0, 255, 0), thickness=-1)
    return merged


def blend_rgb_images(image1, image2, f1=0.5):
    """ Blends to rgb images into a single one. """

    merged = f1*image1.astype(np.float32) + (1.0 - f1)*image2.astype(np.float32)
    merged = merged.round().astype(np.uint8)

    return merged


def blend_rgb_image_mask(image1, mask, f1=0.5):
    """ Blends to rgb images into a single one. """
    m = mask > 0
    # mask = np.round(m.astype(np.float32) * 255)

    merged_both = f1*image1.astype(np.float32) + (1.0 - f1)*mask.astype(np.float32)
    merged = image1.copy()
    merged[m] = merged_both[m]
    merged = merged.round().astype(np.uint8)

    return merged


def blend_scoremap_image(image_gray, scoremap):
    """ Blends scoremaps into the images. Color is identical to what we use for the keypoints"""
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
    colors = colors[:, ::-1]

    # accumulate scoremap info in a colormap
    colormap = np.zeros_like((scoremap[:, :, :3]))
    magmap = np.zeros_like((scoremap[:, :, 0]))
    for kpid in range(scoremap.shape[-1]):
        kp_color = colors[kpid, :]
        mag = np.clip(scoremap[:, :, kpid], 0.0, 1.0)
        mag = np.square(mag) # square makes the plot look nicer
        mask = mag > magmap
        cmap = kp_color.reshape([1, 1, -1]) * np.expand_dims(mag, 2)
        colormap[mask] = cmap[mask]
        magmap[mask] = mag[mask]

    # blend colormap and grayscale image
    colormap *= 255.0
    f = 0.3
    image_gray2 = f*np.expand_dims(image_gray, 2).astype(np.float32) + (1.0 - f)*colormap
    image_gray2 = image_gray2.round().astype(np.uint8)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax1.imshow(image_gray)
    # ax2.imshow(colormap)
    # ax3.imshow(image_gray2)
    # plt.show()

    return image_gray2


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


def plot_hand3d(axis, coords_xyz, vis=None, color_fixed=None, linewidth='1', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
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

    if vis is None:
        vis = np.ones_like(coords_xyz[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i]:
            axis.scatter(coords_xyz[i, 0], coords_xyz[i, 1], coords_xyz[i, 2], marker='o', color=colors[i, :])


def merge_img_scoremap(img, gt, pred, img_type='bgr_norm'):
    img = img.copy()
    gt = gt.copy()
    pred = pred.copy()
    if 'norm' in img_type:
        img = ((img + 0.5)*255).round().astype(np.uint8)
    elif 'float' in img_type:
        img = img.round().astype(np.uint8)

    if 'bgr' in img_type:
        img = img[:, :, ::-1]

    if ('bgr' in img_type) or ('rgb' in img_type):
        img = img.mean(2)

    pred = cv2.resize(pred, img.shape[:2])
    pred = blend_scoremap_image(img, pred)

    gt = cv2.resize(gt, img.shape[:2])
    gt = blend_scoremap_image(img, gt)

    return np.concatenate([gt, pred], 1)


def blend_vecmap_image(image_gray, vecmap):
    """ Blends scoremaps into the images. Color is identical to what we use for the keypoints"""
    if len(image_gray.shape) == 3:
        image_gray = image_gray.mean(2)

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
    colors = colors[:, ::-1]

    vecmap_mag = np.sqrt(np.square(vecmap[:, :, ::2]) + np.square(vecmap[:, :, 1::2]))

    # accumulate scoremap info in a colormap
    colormap = np.zeros_like((vecmap_mag[:, :, :3]))
    magmap = np.zeros_like((vecmap_mag[:, :, 0]))
    for kpid in range(vecmap_mag.shape[2]):
    # for kpid in range(4):
        kp_color = colors[kpid+1, :]
        mag = np.clip(vecmap_mag[:, :, kpid], 0.0, 1.0)
        mask = mag > magmap
        cmap = kp_color.reshape([1, 1, -1]) * np.expand_dims(mag, 2)
        colormap[mask] = cmap[mask]
        magmap[mask] = mag[mask]

    # blend colormap and grayscale image
    colormap *= 255.0
    f = 0.3
    image_gray2 = f*np.expand_dims(image_gray, 2).astype(np.float32) + (1.0 - f)*colormap
    image_gray2 = image_gray2.round().astype(np.uint8)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax1.imshow(image_gray)
    # ax2.imshow(colormap)
    # ax3.imshow(image_gray2)
    # plt.show()

    return image_gray2