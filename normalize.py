import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import resample_img, resample_to_img
from nilearn.datasets import load_mni152_template
from os import walk
from os.path import join
from lib.display import MRI_Image

DATA_LOCATION = "/home/matthew-lee/Data/ADNI"
TEMPLATE_LOCATION = "/home/matthew-lee/Data/ADNI/eva/MNI152_T1_1mm_brain_181x217x181.nii"

def find_all_filetype(location, filetype="nii"):
    approved_files = []
    for (dirpath, dirnames, filenames) in walk(location):
        for f in filenames:
            if f.split(".")[-1] == filetype:
                approved_files.append(join(dirpath, f))
    return approved_files


def normalize(img_location, template_location):
    img = nib.load(img_location)
    print(img)
    template = nib.load(template_location)
    print(template)
    normalized_img = resample_img(img, interpolation="nearest", target_affine=template.affine)
    #normalized_img = resample_to_img(img_location, template)
    return normalized_img, img, template


if __name__ == "__main__":
    files = find_all_filetype(DATA_LOCATION)
    img, original, template = normalize(files[10], TEMPLATE_LOCATION)
    mri_normalized = MRI_Image(img.get_fdata())
    mri_normalized.plot_images()
    mri_normalized = MRI_Image(original.get_fdata())
    mri_normalized.plot_images()

    mri_normalized = MRI_Image(template.get_fdata())
    mri_normalized.plot_images()
    plt.show()

