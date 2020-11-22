import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os


def imread(path, image_res=[128, 128]):
    """
    Utility to read image in RGB format and normalize it
    Parameters:
        path        : type:list. Path to the image to be loaded
        image_res   : type:int list. Array denoting the resized [H,W] of image
    Returns:
        A normalized and resized image
    """
    img = plt.imread(path, format='RGB').astype(np.float)
    img = tf.image.resize(img, image_res).numpy()

    img = img/127.5 - 1.
    return img


def get_samples(path,
                domain='A',
                batch_size=1,
                image_res=[128, 128],
                is_testing=False):
    """
    Method to get a random sample of images
    Parameters:
        path        : type:str. Path to the dataset
        domain: type:str. Domain A or B to pick samples from.
        batch_size  : type:int. Number of images required
        image_res   : type:int list. Array denoting the resized [H,W] of image
        is_testing  : type: bool. Flag to control random flipping
    Returns:
        A list of randomly sampled images
    """
    data_type = "train%s" % domain if not is_testing else "test%s" % domain
    path = glob('{}/{}/*'.format(path, data_type))

    random_sample = np.random.choice(path, size=batch_size)

    imgs = []
    for img_path in random_sample:
        img = imread(img_path, image_res)
        if not is_testing and np.random.random() > 0.5:
            img = np.fliplr(img)
        imgs.append(img)

    return np.array(imgs)


def batch_generator(path,
                    batch_size=1,
                    image_res=[128, 128],
                    is_testing=False):
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
    data_type = "train" if not is_testing else "test"
    path_A = glob('{}/{}A/*'.format(path, data_type))
    path_B = glob('{}/{}B/*'.format(path, data_type))

    num_batches = int(min(len(path_A), len(path_B)) / batch_size)
    num_samples = num_batches * batch_size

    # get num_samples from each domain
    path_A = np.random.choice(path_A, num_samples, replace=False)
    path_B = np.random.choice(path_B, num_samples, replace=False)

    for i in range(num_batches-1):
        batch_A = path_A[i*batch_size:(i+1)*batch_size]
        batch_B = path_B[i*batch_size:(i+1)*batch_size]
        imgs_A, imgs_B = [], []
        for img_A, img_B in zip(batch_A, batch_B):
            img_A = imread(img_A, image_res)
            img_B = imread(img_B, image_res)

            if not is_testing and np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        yield imgs_A, imgs_B


def plot_sample_images(gen_AB,
                       gen_BA,
                       path,
                       epoch=0,
                       batch_num=1,
                       output_dir='maps'):
    """
    Method to plot sample outputs from generator
    Parameters:
        g_AB        :   type:keras model object. Generator model from A->B
        gen_BA      :   type:keras model object. Generator model from B->A
        path        :   type:str. Path to dataset
        epoch       :   type:int. Epoch number, used for output file name
        batch_num   :   type:int. Batch number, used for output file name
        output_dir  :   type:str. Path to save generated output samples
    Returns:
        None
    """
    imgs_A = get_samples(path, domain="A", batch_size=1, is_testing=True)
    imgs_B = get_samples(path, domain="B", batch_size=1, is_testing=True)

    # generate fake samples from both generators
    fake_B = gen_AB.predict(imgs_A)
    fake_A = gen_BA.predict(imgs_B)

    # reconstruct orginal samples from both generators
    reconstruct_A = gen_BA.predict(fake_B)
    reconstruct_B = gen_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B,
                               reconstruct_A,
                               imgs_B, fake_A,
                               reconstruct_B])

    # scale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    os.makedirs(output_dir, exist_ok=True)
    titles = ['Original', 'Translated', 'Reconstructed']

    r, c = 2, 3
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
          axs[i,j].imshow(gen_imgs[cnt])
          axs[i, j].set_title(titles[j])
          axs[i,j].axis('off')
          cnt += 1
    fig.savefig("{}/{}_{}.png".format(output_dir, epoch, batch_num))
    plt.show()
    plt.close()
