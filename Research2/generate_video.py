import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def generate_video1(img)->animation.ArtistAnimation:
    '''
    Utility to generate video out of images (data)
    @Params
        img:list
            list of data with shape (n, h, w) (MAKE SURE THEY ARE FLOATS)
    @Returns
        animation.ArtistAnimation
    '''
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(img)):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    return ani

def generate_video2(plots)->animation.ArtistAnimation:
    '''
    Utility to generate video out of artists
    @Params
        plots:list
            list of plotting artists (MAKE SURE THEY ARE ARTISTS!)
    @Returns
        animation.ArtistAnimation
    '''
    fig = plt.figure()
    ani = animation.ArtistAnimation(plots)
    return ani

if __name__ == '__main__':
    import numpy as np
    fake_data = np.random.randn(10, 10, 10)
    generate_video1(fake_data)
    plt.show()