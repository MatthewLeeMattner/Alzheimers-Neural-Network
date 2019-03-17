import matplotlib.pyplot as plt


def display_image(image, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')


def display_comparison(image_original, image_generated, ax=None):
    if len(ax) is not 2:
        raise ValueError("Ax should be tuple containing two instances of ax")
    if ax is None:
        fig, ax = plt.subplots(ncols=2)
        ax[0].set_title("Original")
        ax[1].set_title("Generated")
    display_image(image_original, ax[0])
    display_image(image_generated, ax[1])


def display_comparison_batch(image_original_batch, image_generated_batch):
    if image_original_batch.shape[0] is not image_generated_batch.shape[0]:
        raise ValueError("Batch not equal")
    batch_size = image_original_batch.shape[0]
    fig, ax = plt.subplots(nrows=batch_size, ncols=2)
    for i in range(batch_size):
        if i is 0:
            ax[i][0].set_title("Original")
            ax[i][1].set_title("Generated")
        display_comparison(image_original_batch[i], image_generated_batch[i], ax[i])
