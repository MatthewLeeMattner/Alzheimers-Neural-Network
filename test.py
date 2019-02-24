from autoencoder import Autoencoder
from data_helper import get_dummy_patches


train, test, val = get_dummy_patches()

encoder = Autoencoder()
model = encoder.model

import matplotlib.pyplot as plt
from display import display_comparison_batch

input_imgs = test[:5]
output = model.predict(input_imgs)

display_comparison_batch(input_imgs[:, :, :, 3], output[:, :, :, 3])
plt.show()

encoder.load_model("1550561880.0557485-autoencoder")
input_imgs = test[:5]
output = model.predict(input_imgs)

display_comparison_batch(input_imgs[:, :, :, 3], output[:, :, :, 3])
plt.show()
