import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    img = np.load("/home/matthew-lee/Data/ADNI/clean/singles/100_train_x.npy")
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[50, :, :])
    axs[1].imshow(img[:, 50, :])
    axs[2].imshow(img[:, :, 50])
    plt.show()
