import os
import cv2
import numpy as np
from random import shuffle
from skimage import transform
from constants import random_transform_args

from constants import face_coverage


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
    ).reshape(new_shape)


def random_warp(image, coverage=face_coverage):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - coverage // 2, 128 + coverage // 2, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)

    mat = transform.SimilarityTransform()
    mat.estimate(src_points, dst_points)
    target_image = cv2.warpAffine(image, mat.params[0:2], (64, 64))
    return warped_image, target_image


def random_transform(image,
                     rotation_range,
                     zoom_range,
                     shift_range,
                     random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(
        image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


def read_image(img_path, random_transform_args=random_transform_args):
    image = cv2.imread(img_path) / 255.0
    image = cv2.resize(image, (256, 256))
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp(image)

    return warped_img, target_img


def minibatch(image_list, batchsize):
    length = len(image_list)
    epoch = i = 0
    shuffle(image_list)
    while True:
        size = batchsize
        if i + size > length:
            shuffle(image_list)
            i = 0
            epoch += 1
        images = np.float32([read_image(image_list[j])
                             for j in range(i, i + size)])
        warped_img, target_img = images[:, 0, :, :, :], images[:, 1, :, :, :]
        i += size
        yield epoch, warped_img, target_img


def minibatchAB(image_list, batchsize):
    batch = minibatch(image_list, batchsize)
    for ep1, warped_img, target_img in batch:
        yield ep1, warped_img, target_img


def get_image_paths(img_dir):
    return [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
