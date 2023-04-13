import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import tqdm
import sys

aor = 4 # angle of repose
pile_shape = (50,50)
def main():
    if len(sys.argv) != 2: events = 3000
    else: events = int(sys.argv[1])
    pile = (np.random.randn(*pile_shape) > 0.5).astype(np.int16) + (np.random.randn(*pile_shape) > 0.5).astype(np.int16) + (np.random.randn(*pile_shape) > 0.5).astype(np.int16)
    pile = pile.astype(np.int16)
    img = []
    avalanche_size = []
    for i in tqdm.tqdm(range(events)):
        x, y = np.random.choice(pile_shape[0]), np.random.choice(pile_shape[1])
        pile[x,y] += 1
        cntr = 0
        while (pile >= aor).any():
            update(pile)
            cntr += 1
            pile[0,:], pile[-1,:], pile[:,0], pile[:,-1] = 0,0,0,0
            img.append(pile.copy())
        avalanche_size.append(cntr)
    display(img, avalanche_size)


def update(pile): # pass by reference
    assert (pile >= aor).any()
    for i in range(1,pile_shape[0]-1):
        for j in range(1,pile_shape[1]-1):
            if pile[i,j] >= aor:
                pile[i,j] = 0
                pile[i+1,j]+=1
                pile[i-1,j]+=1
                pile[i,j-1]+=1
                pile[i,j+1]+=1 
                
def display(img:list, avs:list):

    frames = [] # for storing the generated images
    fig, axs = plt.subplots(1,2)
    for i in tqdm.tqdm(range(len(img))):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])
    print("TOTAL FRAMES:", len(frames))
    ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True,
                                    repeat_delay=1000)
    axs[1] = ani
    axs[0].hist(avs[50**2:], bins = max(avs), log=True)
    #
    plt.show()

if __name__ == '__main__':
    main()