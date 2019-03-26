import numpy as np
import pandas as pd
import utils
import math
from tqdm import tqdm
from os import listdir
from os.path import join

GET_DATASET_SPLIT = True
CREATE_SINGLES = True
BATCH_SINGLES = True
CREATE_PATCHES = True

DATA_DIR = "/home/matthew-lee/Data/ADNI/2Yr_1.5T_norm/"
CSV_INFO = "/home/matthew-lee/Data/ADNI/2Yr_1.5T/" \
    "ADNI1_Complete_2Yr_1.5T_3_17_2019.csv"
DATA_CLEAN = "/home/matthew-lee/Data/ADNI/clean/"

SINGLES_PATH = join(DATA_CLEAN, "singles")
BATCHES_PATH = join(DATA_CLEAN, "batches")
PATCHES_PATH = join(DATA_CLEAN, "patches")
NORMALIZED_PATH = join(DATA_CLEAN, "normalized")

CATEGORIES = ["CN", "MCI", "AD"]

TRAIN_PERCENT = 0.6
BATCH_SIZE = 32
KERNEL = (5, 5, 5)
SLICES = 100

FILE_LIST = utils.get_nifti_files(DATA_DIR)
SUBJECT_LIST = utils.get_subject_list(FILE_LIST)

INFO_DF = pd.read_csv(CSV_INFO)


def image_pipeline(x):
    # TODO: normalize images
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def save_as_numpy_singles(feature, label, name, name_type):
    feature = np.array(feature.dataobj)
    feature = image_pipeline(feature)
    np.save(
        join(SINGLES_PATH, "{}_{}_x.npy".format(name, name_type)), feature
    )
    np.save(
        join(SINGLES_PATH, "{}_{}_y.npy".format(name, name_type)), label
    )


def merge_numpy_singles(x_file_list, y_file_list):
    features = []
    labels = []
    for x, y in zip(x_file_list, y_file_list):
        feature = np.load(join(DATA_CLEAN, SINGLES_PATH, x))
        label = np.load(join(DATA_CLEAN, SINGLES_PATH, y))
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)


def merge_numpy_batch(x_file_list, y_file_list, name):
    for i in range(math.ceil(len(x_file_list) / BATCH_SIZE)):
        if i*BATCH_SIZE + BATCH_SIZE < len(x_file_list):
            batch_x = x_file_list[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
            batch_y = y_file_list[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
        else:
            batch_x = x_file_list[i * BATCH_SIZE:]
            batch_y = y_file_list[i * BATCH_SIZE:]
        batch_x, batch_y = merge_numpy_singles(batch_x, batch_y)
        np.save(join(BATCHES_PATH, "{}_{}_x.npy".format(name, i)), batch_x)
        np.save(join(BATCHES_PATH, "{}_{}_y.npy".format(name, i)), batch_y)


########################################
#           SPLIT DATASET
########################################
if GET_DATASET_SPLIT:
    # Setup train test val split
    print("Splitting dataset")
    dataset = utils.train_test_val_by_subject(TRAIN_PERCENT,
                                              SUBJECT_LIST,
                                              CATEGORIES,
                                              FILE_LIST,
                                              DATA_DIR,
                                              INFO_DF)
    train_x, train_y, test_x, test_y, val_x, val_y = dataset

    print(len(train_x), len(train_y))
    print(len(test_x), len(test_y))
    print(len(val_x), len(val_y))

########################################
#          CREATE SINGLE IMAGES
########################################
if CREATE_SINGLES:
    print("Creating singles")
    print("Training singles:")
    iterator = 0
    for feature, label in zip(tqdm(train_x), train_y):
        save_as_numpy_singles(feature, label, iterator, "train")
        iterator += 1

    print("Testing singles:")
    iterator = 0
    for feature, label in zip(tqdm(test_x), test_y):
        save_as_numpy_singles(feature, label, iterator, "test")
        iterator += 1

    print("Validation singles:")
    iterator = 0
    for feature, label in zip(tqdm(val_x), val_y):
        save_as_numpy_singles(feature, label, iterator, "val")
        iterator += 1


########################################
#            BATCH IMAGES
########################################
if BATCH_SINGLES:
    single_files = listdir(join(DATA_CLEAN, SINGLES_PATH))
    train_files_x = []
    test_files_x = []
    val_files_x = []

    train_files_y = []
    test_files_y = []
    val_files_y = []

    print("Batching singles:")
    for f in tqdm(single_files):
        _, f_type, f_type_usage = f.split("_")
        f_type_usage = f_type_usage.split(".")[0]
        if f_type == "train":
            if f_type_usage == "x":
                train_files_x.append(f)
            else:
                train_files_y.append(f)
        elif f_type == "test":
            if f_type_usage == "x":
                test_files_x.append(f)
            else:
                test_files_y.append(f)
        elif f_type == "val":
            if f_type_usage == "x":
                val_files_x.append(f)
            else:
                val_files_y.append(f)

    print(len(train_files_x))
    print(len(test_files_x))
    print(len(val_files_x))
    print(len(train_files_y))
    print(len(test_files_y))
    print(len(val_files_y))

    # val_images = merge_numpy_singles(val_files)
    # print(val_images.shape)
    # test_images = merge_numpy_singles(test_files)
    # print(test_images.shape)
    merge_numpy_batch(train_files_x, train_files_y, "train")
    merge_numpy_batch(test_files_x, test_files_y, "test")
    merge_numpy_batch(val_files_x, val_files_y, "val")


########################################
#            CREATE PATCHES
########################################
if CREATE_PATCHES:
    print("Creating patches:")
    batch_files = listdir(join(DATA_CLEAN, BATCHES_PATH))

    train_iter = 0
    test_iter = 0
    val_iter = 0
    for f in tqdm(batch_files):
        b_type, b_index, b_usage = f.split("_")
        b_usage = b_usage.split(".")[0]
        if b_usage == "x":
            batch = np.load(join(DATA_CLEAN, BATCHES_PATH, f))
            patches = utils.slice_random_patches_batch(batch, KERNEL, SLICES)

            if b_type == "train":
                np.save(
                    join(DATA_CLEAN, PATCHES_PATH,
                         "train_{}.npy".format(train_iter)),
                    patches
                )
                train_iter += 1
            elif b_type == "test":
                np.save(
                    join(DATA_CLEAN, PATCHES_PATH,
                         "test_{}.npy".format(test_iter)),
                    patches
                )
                test_iter += 1
            elif b_type == "val":
                np.save(
                    join(DATA_CLEAN, PATCHES_PATH,
                         "val_{}.npy".format(val_iter)),
                    patches
                )
                val_iter += 1
