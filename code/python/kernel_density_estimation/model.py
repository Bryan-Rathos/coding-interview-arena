"""
Model for training and testing on MNIST and CIFAR.
"""
import numpy as np

import mnist_loader
import cifar_loader
import kernel_density_estimator as kde

train_num = 10000
val_end = 20000
visualize_mnist = False
visualize_cifar = False

(x_train, t_train), (x_test, t_test) = mnist_loader.load_mnist(normalize=True,
                                                               flatten=True)

x_val = x_train[train_num:val_end]
x_train = x_train[0:train_num]
sigma = 0.2

short_batch = 50

mean = kde.model(x_train[0:short_batch], x_val[0:short_batch], sigma)

print('Mean kde ', mean)
mean = kde.sklearn_kde(x_train[0:50], x_val[0:50], sigma)

print('Mean sklearn kde ', mean)


if visualize_mnist:
    # Show the sample image
    rand_idx = np.random.randint(0, train_num - 1)
    img = x_train[rand_idx]
    label = t_train[rand_idx]

    img = img.reshape(28, 28)

    mnist_loader.img_show(img, label)


# CIFAR
images_train, cls_train, images_test, cls_test = cifar_loader.load_cifar()

x_val = images_train[train_num:val_end]
x_train = images_train[0:train_num]
sigma = 0.2

short_batch = 50

x_train = x_train.reshape((-1, 32*32 * 3))
x_val = x_val.reshape((-1, 32*32 * 3))


mean = kde.model(x_train[0:short_batch], x_val[0:short_batch], sigma)

print('Mean kde cifar ', mean)

mean = kde.sklearn_kde(x_train[0:50], x_val[0:50], sigma)

print('Mean sklearn kde cifar ', mean)


if visualize_cifar:
    # Get the first images from the test-set.
    images = images_test[0:9]

    # Get the true classes for those images.
    cls_true = cls_test[0:9]

    # Plot the images and labels using our helper-function above.
    cifar_loader.plot_images(images=images, cls_true=cls_true, smooth=False)
