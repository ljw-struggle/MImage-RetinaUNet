# -*- coding: utf-8 -*-
from skimage import morphology,data,color
import matplotlib.pyplot as plt

def skeletonize():
    image=color.rgb2gray(data.horse())
    image=1-image # Invert
    skeleton =morphology.skeletonize(image)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax1.imshow(image, cmap=plt.cm.get_cmap('gray'))
    ax1.set_title('original', fontsize=15)
    ax1.axis('off')
    ax2.imshow(skeleton, cmap=plt.cm.get_cmap('gray'))
    ax2.set_title('skeleton', fontsize=15)
    ax2.axis('off')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    skeletonize()