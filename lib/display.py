import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

class MRI_Image:

    def __init__(self, image):
        self.image = image
        self.image_transposed = image.T

    def flip_z(self):
        self.image = np.flip(self.image, 2)
        self.image_transposed = self.image.T

    def flip_y(self):
        self.image = np.flip(self.image, 1)
        self.image_transposed = self.image.T

    def flip_x(self):
        self.image = np.flip(self.image, 0)
        self.image_transposed = self.image.T

    def update_image(self, val):
        image_slice = int(val) -1
        self.im.set_data(self.image[image_slice])
        self.fig.canvas.draw_idle()

    def update_image_transposed(self, val):
        image_slice = int(val) -1
        self.im_t.set_data(self.image_transposed[image_slice])
        self.fig.canvas.draw_idle()

    def plot_image(self):
        self.im = self.ax.imshow(self.image[int(self.image.shape[0]/2)])
        ax_slice = plt.axes([0.55, 0.02, 0.35, 0.03])
        self.ax.set_aspect(1)
        self.image_slider = Slider(ax_slice, "", 1, int(self.image.shape[0]),  valstep=1, valfmt='%.0f',
                                  valinit=int(self.image.shape[0]/2))
        self.image_slider.on_changed(self.update_image)

    def plot_image_transposed(self):
        self.im_t = self.ax_t.imshow(self.image_transposed[int(self.image_transposed.shape[0]/2)])
        ax_slice = plt.axes([0.125, 0.025, 0.35, 0.03])
        self.image_t_slider = Slider(ax_slice, "", 1, int(self.image_transposed.shape[0]),  valstep=1, valfmt='%.0f',
                                    valinit=int(self.image_transposed.shape[0]/2))
        self.image_t_slider.on_changed(self.update_image_transposed)

    def plot_images(self):
        self.fig, ax = plt.subplots(ncols=2)
        self.fig.set_size_inches(9, 5, forward=True)
        self.ax_t, self.ax = ax
        self.plot_image_transposed()
        self.plot_image()
