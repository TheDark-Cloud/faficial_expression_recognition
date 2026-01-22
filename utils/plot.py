import matplotlib.pyplot as plt
import numpy as np

size = (15, 10)

def sample_plot(data, n):
    for image, labes in data.take(1):
        plt.figure(figsize=size) # setting the size of the plot
        for i in range(n):
            ax = plt.subplot(3, n, i + 1) # Allowing multiples plot on the same line

            # setting the value range to an acceptable range
            img = np.clip(image[i].numpy(), 0, 255).astype(np.uint8)
            plt.imshow(img) # plotting the image
            class_value = data.class_names[labes[i]]
            plt.title(f'Class: {class_value}')
            plt.axis('off')
        plt.show()

def train_plot(data):
    pass

