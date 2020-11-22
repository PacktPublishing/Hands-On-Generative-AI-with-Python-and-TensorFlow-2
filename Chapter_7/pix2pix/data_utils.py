import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os


def imread(path):
    """
    Utility to read image in RGB format, split into two halves and normalize
    Parameters:
        path: type:list. Path to the image to be loaded
    Returns:
        A tuple consisting of two halves of the input image
    """
    img = plt.imread(path, format='RGB').astype(np.float)
    h, w, _ = img.shape
    half_w = int(w/2)
    return (img[:, :half_w, :]/127.5)-1, (img[:, half_w:, :]/127.5)-1


def _get_images(image_path_list, img_res=[256, 256], is_testing=False):
    """
    Utility to process a list of images, resize and randomly perform
    horizontal flips
    Parameters:
        image_path_list: type:list. List of paths of images to be processed
        image_res: type:int list. Array denoting the resized [H,W] of image
        is_testing: type: bool. Flag to control random flipping
    Returns:
        A tuple of two lists (source, target)
    """
    imgs_source = []
    imgs_target = []
    for img_path in image_path_list:
        img_src, img_tgt = imread(img_path)
        img_src = tf.image.resize(img_src, img_res).numpy()
        img_tgt = tf.image.resize(img_tgt, img_res).numpy()

        # Flip random images horizontally during training
        if not is_testing and np.random.random() < 0.5:
            img_src = np.fliplr(img_src)
            img_tgt = np.fliplr(img_tgt)

        imgs_source.append(img_src)
        imgs_target.append(img_tgt)
    imgs_source = np.array(imgs_source)
    imgs_target = np.array(imgs_target)

    return imgs_source, imgs_target


def get_samples(path, batch_size=1, img_res=[256, 256], is_testing=False):
    """
    Method to get a random sample of images
    Parameters:
        path: type:str. Path to the dataset
        batch_size: type:int. Number of images required
        image_res: type:int list. Array denoting the resized [H,W] of image
        is_testing: type: bool. Flag to control random flipping
    Returns:
        A tuple of two lists (source, target) with randomly sampled images
    """
    data_type = "train" if not is_testing else "val"
    path = glob('{}/{}/*'.format(path, data_type))

    random_sample = np.random.choice(path, size=batch_size)
    return _get_images(random_sample, img_res, is_testing)


def batch_generator(path, batch_size=1, img_res=[256, 256], is_testing=False):
    """
    Method to generate batch of images
    Parameters:
        path: type:str. Path to the dataset
        batch_size: type:int. Number of images required
        image_res: type:int list. Array denoting the resized [H,W] of image
        is_testing: type: bool. Flag to control random flipping
    Returns:
        yields a tuple of two lists (source,target)
    """
    data_type = "train" if not is_testing else "val"
    path = glob('{}/{}/*'.format(path, data_type))

    num_batches = int(len(path) / batch_size)
    for i in range(num_batches-1):
        batch = path[i*batch_size:(i+1)*batch_size]
        imgs_source, imgs_target = _get_images(batch, img_res, is_testing)
        yield imgs_source, imgs_target


def plot_sample_images(generator,
                       path,
                       epoch=0,
                       batch_num=1,
                       output_dir='maps'):
    """
    Method to plot sample outputs from generator
    Parameters:
        generator:  type:keras model object. Generator model
        path:       type:str. Path to dataset
        epoch:      type:int. Epoch number, used for output file name
        batch_num:  type:int. Batch number, used for output file name
        output_dir: type:str. Path to save generated output samples
    Returns:
        None
    """

    imgs_source, imgs_target = get_samples(path,
                                           batch_size=3,
                                           img_res=[256, 256],
                                           is_testing=True)
    fake_imgs = generator.predict(imgs_target)

    gen_imgs = np.concatenate([imgs_target, fake_imgs, imgs_source])

    # scale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    os.makedirs(output_dir, exist_ok=True)

    titles = ['Condition', 'Generated', 'Original']
    rows, cols = 3, 3
    fig, axs = plt.subplots(rows, cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("{}/{}_{}.png".format(output_dir, epoch, batch_num))
    plt.show()
    plt.close()
