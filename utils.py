import os
from os.path import join
import numpy as np
import nibabel as nib
import random
from random import randint


def get_nifti_files(data_dir):
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


def get_dummy_patches():
    x_train = np.random.randn(10000, 5, 5, 5)
    x_test = np.random.randn(10000, 5, 5, 5)
    x_val = np.random.randn(10000, 5, 5, 5)
    return x_train, x_test, x_val


def load_nifti(location):
    return nib.load(location)


def get_img_data(img_location, info_df):
    img_id = extract_img_id(img_location)
    return info_df[info_df['Image Data ID'] == int(img_id)]


def get_label(img_location, info_df):
    row = get_img_data(img_location, info_df)
    return list(row['Group'])[0]


def get_one_hot_encoding(indexes):
    encodings = np.zeros((len(indexes), indexes.max()+1))
    encodings[np.arange(indexes.size), indexes] = 1
    return encodings


def get_label_indexes(label, categories):
    try:
        return categories.index(label)
    except ValueError:
        return None


def group_by_subject(info_df):
    pass


def load_by_subject_id(subject_id,
                       subject_list,
                       file_list,
                       data_dir,
                       info_df):
    images = []
    labels = []
    for subject, location in zip(subject_list, file_list):
        if subject == subject_id:
            images.append(load_nifti(join(data_dir, location)))
            labels.append(get_label(location, info_df))
    return images, labels


def load_by_subject_ids(subject_ids,
                        subject_list,
                        file_list,
                        data_dir,
                        info_df):
    images = []
    labels = []
    for subject_id in subject_ids:
        image, label = load_by_subject_id(subject_id,
                                          subject_list,
                                          file_list,
                                          data_dir,
                                          info_df)
        images.append(image)
        labels.append(label)
    return images, labels


def slice_patch(img, coords, kernel):
    if len(kernel) < 3:
        raise ValueError("slice_patch: kernel shape"
                         "needs to be 3, not {}".format(kernel.shape))
    if len(coords) < 3:
        raise ValueError("slice_patch: coords "
                         "shape needs to be 3, not {}".format(coords.shape))

    x, y, z = coords
    width, height, depth = img.shape
    k_w, k_h, k_d = kernel
    patch = img[x:x+k_w, y:y+k_h, z:z+k_d]
    return patch


def slice_random_patches(img, kernel, slices=100):
    try:
        if len(kernel) < 3:
            raise ValueError("slice_patch: kernel shape"
                             "needs to be 3, not {}".format(kernel.shape))
        if type(slices) is not int:
            raise TypeError("Slices needs to be an integer")
        if slices < 1:
            raise ValueError("Slices must be >= 1. "
                             "Found {} instead".format(slices))
        k_w, k_h, k_d = kernel
        width, height, depth = img.shape
        patches = []
        for i in range(slices):
            x = randint(0, width - k_w)
            y = randint(0, height - k_h)
            z = randint(0, depth - k_d)
            patch = slice_patch(img, (x, y, z), kernel)
            patches.append(patch)
    except ValueError:
        print("Error on image. Image shape: {}".format(img.shape))
        print(img)
    return np.array(patches)


def slice_random_patches_batch(batch, kernel, slices=100):
    patches = []
    for img in batch:
        p = slice_random_patches(img, kernel, slices)
        patches.extend(p)
    return np.array(patches)


def flatten_list(l):
    flattened = [item for sublist in l for item in sublist]
    return np.array(flattened)


def paired_shuffle(list_1, list_2):
    combined = list(zip(list_1, list_2))
    random.shuffle(combined)
    list_1[:], list_2[:] = zip(*combined)
    return list_1, list_2


def train_test_val_by_subject(train_percent,
                              subjects,
                              categories,
                              file_list,
                              data_dir,
                              info_df):

    # Get all the features(images) and labels by unique subject ids
    unique_subjects = get_unique_subjects(subjects)
    features, labels = load_by_subject_ids(unique_subjects,
                                           subjects,
                                           file_list,
                                           data_dir,
                                           info_df)

    # Shuffle by subject groupings
    features, labels = paired_shuffle(features, labels)

    # Train split by subject groupings
    split_index = int(len(features) * train_percent)

    train_x, train_y = features[:split_index], labels[:split_index]
    test_x, test_y = features[split_index:], labels[split_index:]
    val_split = len(test_x) // 2
    val_x = test_x[val_split:]
    val_y = test_y[val_split:]
    test_x = test_x[:val_split]
    test_y = test_y[:val_split]

    # Remove subject groupings
    train_x = flatten_list(train_x)
    train_y = flatten_list(train_y)
    test_x = flatten_list(test_x)
    test_y = flatten_list(test_y)
    val_x = flatten_list(val_x)
    val_y = flatten_list(val_y)

    # One hot encode the y values
    train_y = get_one_hot_encoding(
        np.array([get_label_indexes(x, categories) for x in train_y])
    )
    test_y = get_one_hot_encoding(
        np.array([get_label_indexes(x, categories) for x in test_y])
    )
    val_y = get_one_hot_encoding(
        np.array([get_label_indexes(x, categories) for x in val_y])
    )

    # Shuffle again with seperation performed
    train_x, train_y = paired_shuffle(train_x, train_y)
    test_x, test_y = paired_shuffle(test_x, test_y)
    val_x, val_y = paired_shuffle(val_x, val_y)

    # Return the train, test and val set
    return train_x, train_y, test_x, test_y, val_x, val_y


if __name__ == "__main__":
    img_location = "/home/matthew-lee/Data/ADNI/2Yr_1.5T_norm/"\
        "ADNI_005_S_0546_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_"\
        "20080318131759141_S46353_I98738_MNI.nii"
    train_test_val_by_subject(0.7)
