from PIL import Image
from matplotlib import pyplot as plt

def crop_images(image, save_path, prefix, save = False):

    images = list()

    images.append(image.crop((0, 0, 256, 256)))
    images.append(image.crop((256, 0, 512, 256)))
    images.append(image.crop((512, 0, 768, 256)))
    images.append(image.crop((768, 0, 1024, 256)))
    images.append(image.crop((1024, 0, 1280, 256)))
    images.append(image.crop((1280, 0, 1536, 256)))
    images.append(image.crop((1536, 0, 1792, 256)))
    images.append(image.crop((1664, 0, 1920, 256)))

    images.append(image.crop((0, 256, 256, 512)))
    images.append(image.crop((256, 256, 512, 512)))
    images.append(image.crop((512, 256, 768, 512)))
    images.append(image.crop((768, 256, 1024, 512)))
    images.append(image.crop((1024, 256, 1280, 512)))
    images.append(image.crop((1280, 256, 1536, 512)))
    images.append(image.crop((1536, 256, 1792, 512)))
    images.append(image.crop((1664, 256, 1920, 512)))

    images.append(image.crop((0, 512, 256, 768)))
    images.append(image.crop((256, 512, 512, 768)))
    images.append(image.crop((512, 512, 768, 768)))
    images.append(image.crop((768, 512, 1024, 768)))
    images.append(image.crop((1024, 512, 1280, 768)))
    images.append(image.crop((1280, 512, 1536, 768)))
    images.append(image.crop((1536, 512, 1792, 768)))
    images.append(image.crop((1664, 512, 1920, 768)))

    images.append(image.crop((0, 768, 256, 1024)))
    images.append(image.crop((256, 768, 512, 1024)))
    images.append(image.crop((512, 768, 768, 1024)))
    images.append(image.crop((768, 768, 1024, 1024)))
    images.append(image.crop((1024, 768, 1280, 1024)))
    images.append(image.crop((1280, 768, 1536, 1024)))
    images.append(image.crop((1536, 768, 1792, 1024)))
    images.append(image.crop((1664, 768, 1920, 1024)))

    if save:
        for i, image in enumerate(images):
            print(save_path + "image_{0}_{1}.png".format(prefix, i))
            image.save(save_path + "image_{0}_{1}.png".format(prefix, i))

    return images


def normalize_images(path):
    path_prefix = "train/"
    for idx in range(520):
        if idx == 350:
            path_prefix = "test/"
        elif idx == 400:
            path_prefix = "val/"
        mask_path = "/home/martin/Desktop/arm/train/mask/capture_{0}_mask.png".format(idx)
        image1 = Image.open(mask_path)
        new_masked_images = crop_images(image1, "/home/martin/Desktop/arm/data/mask/", idx)
        origin_path = "/home/martin/Desktop/arm/train/original/capture_{0}_original.png".format(idx)
        image2 = Image.open(origin_path)
        new_original_image = crop_images(image2, "/home/martin/Desktop/arm/data/original/", idx)

        for idx2, (i, j) in enumerate(zip(new_masked_images, new_original_image)):
            new_iamge = Image.new('RGB', (2*i.size[0], i.size[1]), (250, 250, 250))
            new_iamge.paste(i, (0,0))
            new_iamge.paste(j, (i.size[0], 0))
            new_iamge.save(path + path_prefix + "image_{0}_{1}.png".format(idx, idx2))


if __name__ == "__main__":
    normalize_images("/home/martin/Desktop/arm/data/merged/")