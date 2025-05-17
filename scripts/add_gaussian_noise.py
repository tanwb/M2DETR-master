import os
from PIL import Image
import numpy as np


def add_gaussian_noise(image, mean, var, noise_intensity=1):
    """
    Add Gaussian noise to an image
    :param image: PIL Image object
    :param mean: Mean of the noise
    :param var: Standard deviation of the noise
    :param noise_intensity: Noise intensity, range between 0 and 1
    :return: Image with Gaussian noise
    """
    image_np = np.array(image)
    noise = np.random.normal(mean, var, image_np.shape) * noise_intensity
    noisy_image = image_np + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def process_images_in_folder_train(folder_path_train_in, folder_path_train_out, mean, var):
    if not os.path.exists(folder_path_train_out):
        os.makedirs(folder_path_train_out)
        print("Folder created")
    else:
        print("Folder already exists")

    for filename in os.listdir(folder_path_train_in):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(folder_path_train_in, filename)  # Fixed variable name
            with Image.open(file_path) as img:
                noisy_img = add_gaussian_noise(img, mean=mean, var=var, noise_intensity=1)
                noisy_img.save(os.path.join(folder_path_train_out, f'{filename}'))


def process_images_in_folder_test(folder_path_test_in, folder_path_test_out, mean, var):
    if not os.path.exists(folder_path_test_out):
        os.makedirs(folder_path_test_out)
        print("Folder created")
    else:
        print("Folder already exists")

    for filename in os.listdir(folder_path_test_in):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(folder_path_test_in, filename)
            with Image.open(file_path) as img:
                noisy_img = add_gaussian_noise(img, mean=mean, var=var, noise_intensity=1)
                noisy_img.save(os.path.join(folder_path_test_out, f'{filename}'))


if __name__ == '__main__':
    folder_path_train_in = 'XXX/data_ip102/coco_format/train'
    folder_path_train_out = 'XXX/data_ip102/coco_format/noise/train_add_gauss_noise'

    folder_path_test_in = 'XXX/data_ip102/coco_format/test'
    folder_path_test_out = 'XXX/data_ip102/coco_format/noise/test_add_gauss_noise'

    mean = 0  # Mean of the noise
    var = 20  # Standard deviation of the noise

    process_images_in_folder_train(folder_path_train_in, folder_path_train_out, mean=mean, var=var)
    process_images_in_folder_test(folder_path_test_in, folder_path_test_out, mean=mean, var=var)
