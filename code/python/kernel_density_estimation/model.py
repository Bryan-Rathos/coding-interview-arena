"""
Model for training and testing on MNIST and CIFAR.
"""
import numpy as np

import mnist_loader
#import kernel_density_estimator as kde

train_num = 10000
visualize_rand = True
(x_train, t_train), (x_test, t_test) = mnist_loader.load_mnist(normalize=False,
                                                               flatten=True)

x_train = x_train[0:train_num]
x_val = x_train[train_num:20000]


if visualize_rand:
    # Show the sample image
    rand_idx = np.random.randint(0, train_num - 1)
    img = x_train[rand_idx]
    label = t_train[rand_idx]

    img = img.reshape(28, 28)

    mnist_loader.img_show(img, label)
