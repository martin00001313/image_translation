import core
from matplotlib import  pyplot as plt

if __name__ == "__main__":
    inp, re = core.load("/home/martin/Desktop/arm/train/merged_norm/train/image_0.png")

    # check loaded objects
    plt.figure()
    plt.imshow(inp/255.)
    plt.figure()
    plt.imshow(re/255.)
    plt.show()

    # random crop the images
    plt.figure(figsize=(6, 6))
    for i in range(0):
        rj_inp, rj_re = core.random_jitter(inp, re)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis('off')