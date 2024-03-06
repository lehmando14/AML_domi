import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML


def show_video(video: np.ndarray) -> None:
    '''
    the video has to have the following dimensions (height, width, frames)
    '''
    fig, ax = plt.subplots()

    #Display the first frame (initialization)
    img = ax.imshow(video[:, :, 0], cmap='gray', vmin=0, vmax=255)

    # Animation update function
    def update(frame):
        img.set_array(video[:, :, frame])
        return [img]

    # Set up the animation
    animation = FuncAnimation(fig, update, frames=video.shape[2], blit=True)

    # Display the animation using HTML
    display(HTML(animation.to_jshtml()))

def show_img(img: np.ndarray) -> None:
    '''
    the img has dimensions (height, width)
    '''
    plt.imshow(img, cmap='gray')
    plt.show()