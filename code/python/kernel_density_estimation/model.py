"""
Model for training and testing on MNIST and CIFAR.
"""
import numpy as np
import time

import mnist_loader
import cifar_loader
import kernel_density_estimator as kde

# Number of training samples
train_num = 10000
# Remainder used for validation
val_end = 20000

visualize_mnist = False
visualize_cifar = False

run_mnist = True
run_cifar = True

dataset = 'cifar'


if dataset == 'mnist':

    ####################################################################
    # MNIST
    ####################################################################
    (x_train, t_train), (x_test, t_test) = mnist_loader.load_mnist(
        normalize=True, flatten=True)

    if run_mnist:

        x_val = x_train[train_num:val_end]
        x_train = x_train[0:train_num]

        sigma_params = [0.05, 0.08, 0.1, 0.2, 0.5, 1., 1.5, 2]

        for sigma in sigma_params:

            start = time.time()

            mean = kde.model(x_train, x_test, sigma)

            print('KDE took -> ', time.time() - start)
            print('Mean kde ', mean, ' with sigma ', 0.2)

            """
            start = time.time()
            mean = kde.sklearn_kde(x_train, x_val, sigma)
            print('Sklearn KDE took -> ', time.time() - start)
            print('Mean sklearn kde ', mean)
            """

    if visualize_mnist:
        # Show image grid
        img = mnist_loader.visualize_mnist_grid(x_train, n=20,
                                                data_img_rows=28, data_img_cols=28,
                                                data_img_channels=1)

        # Show one sample image
        """
        rand_idx = np.random.randint(0, train_num - 1)
        img = x_train[rand_idx]
        label = t_train[rand_idx]
        img = img.reshape(28, 28)
        mnist_loader.img_show(img, label)
        """

elif dataset == 'cifar':

    # CIFAR
    images_train, cls_train, images_test, cls_test = cifar_loader.load_cifar()

    ####################################################################
    # CIFAR
    ####################################################################
    if run_cifar:

        x_val = images_train[train_num:val_end]
        x_train = images_train[0:train_num]

        x_train = x_train.reshape((-1, 32*32*3))
        x_val = x_val.reshape((-1, 32*32*3))
        x_test = images_test.reshape((-1, 32*32*3))

        sigma_params = [0.05, 0.08, 0.1, 0.2, 0.5, 1.]

        for sigma in sigma_params:

            start = time.time()
            mean = kde.model(x_train, x_test, sigma)
            print('KDE took -> ', time.time() - start)
            print('Mean kde cifar ', mean)
            print('Sigma ', sigma)


    if visualize_cifar:
        # Get the first images from the test-set.
        images = images_test[0:9]

        # Get the true classes for those images.
        cls_true = cls_test[0:9]

        # Plot the images and labels using our helper-function above.
        cifar_loader.plot_images(images=images, cls_true=cls_true, smooth=False)
