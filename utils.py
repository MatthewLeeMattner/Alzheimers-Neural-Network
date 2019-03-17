import os
from os.path import join
import numpy as np
import nibabel as nib
import pandas as pd
from random import randint

DATA_DIR = "/home/matthew-lee/Data/ADNI/2Yr_1.5T_norm/"
CSV_INFO = "/home/matthew-lee/Data/ADNI/2Yr_1.5T/ADNI1_Complete_2Yr_1.5T_3_17_2019.csv"

INFO_DF = pd.read_csv(CSV_INFO)

def get_nifti_files(data_dir=DATA_DIR):
    file_list = []
    files = os.listdir(data_dir)
    for f in files:
        if f.split(".")[-1] == "nii":
            file_list.append(f)
    return file_list

def get_subject_list(file_list):
    subject_list = []
    for f in file_list:
        subject_id = extract_subject_id(f)
        subject_list.append(subject_id)
    return subject_list

def get_unique_subjects(subject_list):
    return set(subject_list)

def extract_subject_id(img_location):
    split_location = img_location.split("/")[-1]
    subject_id = "_".join(split_location.split("_")[1:4])
    return subject_id

def extract_img_id(img_location):
    split_location = img_location.split("/")[-1]
    img_id = split_location.split("_")[-2][1:]
    return img_id

FILE_LIST = get_nifti_files()
SUBJECT_LIST = get_subject_list(FILE_LIST)


def get_dummy_patches():
    x_train = np.random.randn(10000, 5, 5, 5)
    x_test = np.random.randn(10000, 5, 5, 5)
    x_val = np.random.randn(10000, 5, 5, 5)
    return x_train, x_test, x_val

def load_nifti(location):
    return nib.load(location)

def get_img_data(img_location, info_df=INFO_DF):
    img_id = extract_img_id(img_location)
    return info_df[info_df['Image Data ID'] == int(img_id)]

def get_label(img_location):
    row = get_img_data(img_location)
    return row['Group']

def group_by_subject(info_df=INFO_DF):
    pass

def load_by_subject_id(subject_id, subject_list=SUBJECT_LIST, file_list=FILE_LIST, data_dir=DATA_DIR, info_df=INFO_DF):
    print(subject_id)
    images = []
    for subject, location in zip(subject_list, file_list):
        if subject == subject_id:
            images.append(load_nifti(join(data_dir, location)))
    return images

def load_by_subject_ids(subject_ids):
    images = []
    for subject_id in subject_ids:
        images.append(load_by_subject_id(subject_id))
    return images

def slice_patch(img, kernel):
    if len(kernel) < 3:
        raise ValueError("slice_patch: kernel shape needs to be 3, not {}".format(kernel.shape))
    width, height, depth = img.shape
    k_w, k_h, k_d = kernel
    x = randint(0, width - k_w)
    y = randint(0, height - k_h)
    z = randint(0, depth - k_d)
    patch = img[x:x+k_w, y:y+k_h, z:z+k_d]
    return patch


if __name__ == "__main__":
    img_location = "/home/matthew-lee/Data/ADNI/2Yr_1.5T_norm/ADNI_005_S_0546_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080318131759141_S46353_I98738_MNI.nii"
    unique_subjects = get_unique_subjects(SUBJECT_LIST)
    print(unique_subjects)
    images = load_by_subject_ids(unique_subjects)
    image = images[0][0].get_fdata()

    from lib.display import MRI_Image
    import matplotlib.pyplot as plt
    display_mri = MRI_Image(image)
    display_mri.plot_images()
    plt.show()
