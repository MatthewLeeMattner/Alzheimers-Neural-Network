import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def get_crop_dimensions(image):
    result = np.argwhere(image)
    y_min, x_min, z_min = result.min(axis=0)
    y_max, x_max, z_max = result.max(axis=0)
    return [y_min, y_max], [x_min, x_max], [z_min, z_max]


def crop_image(image, x, y, z):
    return image[x[0]:x[1], y[0]:y[1], z[0]:z[1]]


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)


def subtract_mean_divide_std(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std


def data_pipeline(image, template):
    x, y, z = get_crop_dimensions(template)
    image = crop_image(image, x, y, z)
    image = subtract_mean_divide_std(image)
    return image


if __name__ == "__main__":
    template = nib.load("/home/matthew-lee/Data/ADNI/"
                        "eva/MNI152_T1_1mm_brain_181x217x181.nii")
    image = nib.load("/home/matthew-lee/Data/ADNI/2Yr_1.5T_norm/"
                     "ADNI_010_S_0067_MR_MPR____N3__Scaled_Br_20"
                     "070828102413728_S25341_I70630_MNI.nii")
    template_np = template.get_fdata()
    image_np = template.get_fdata()

    # x, y, z = get_crop_dimensions(template_np)
    # template_cropped = crop_image(template_np, x, y, z)
    # template_cropped = normalize_image(template_cropped)
    # print(template_cropped[50, 50, :])

    image = data_pipeline(image_np, template_np)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(image_np[50, :, :])
    axs[0, 1].imshow(image_np[:, 50, :])
    axs[0, 2].imshow(image_np[:, :, 50])

    axs[1, 0].imshow(image[50, :, :])
    axs[1, 1].imshow(image[:, 50, :])
    axs[1, 2].imshow(image[:, :, 50])
    plt.show()
