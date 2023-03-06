'''my own hopfield network'''
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

side_len = 4*2
def gen_target_state() -> np.ndarray:
    '''Make a flattened rectangle to memorize'''
    target_state = np.ones((side_len,side_len), dtype = np.short)
    target_state[np.random.randint(0, side_len//2 - 1): np.random.randint(side_len//2 + 1, side_len),
                 np.random.randint(0, side_len//2 - 1): np.random.randint(side_len//2 + 1, side_len)]\
                = -1
    target_state = target_state.reshape((-1,))
    return target_state

def gen_target_states(num:int) -> np.ndarray:
    '''MULTIPLE target states in a array'''
    assert isinstance(num, int) or isinstance(num, np.ScalarType)
    return np.array([gen_target_state() for i in range(num)])

def gen_init_state(type:str = None) -> np.ndarray:
    '''Args:
        type:
            -"random"
            - "masked"
            '''
    if type is None or type.lower() == "random":
        state = np.random.rand(side_len**2)
        state = (state > 0.5) * 2 - 1
        state = state.astype(np.short)
        return state
    elif type.lower() == "masked":
        state = gen_target_state()
        state[len(state)//2:] = -1
        state = state.astype(np.short)
        return state

def compute_energy(wm:np.ndarray, state:np.ndarray, version:int = 0 ):
    if version == 0:
        # for i in range(wm.shape[0]):
        #     for j in range(wm.shape[1]):
        #         energy += wm[i,j]*state[i]*state[j]
        energy = state.T @ wm @ state
        energy = energy * -.5
        for i in range(len(state)):
            energy += 0 * state[i] #does nothing
            
        return energy

    else: return compute_energy(wm, state, 0)

def main():
    num_ts = 10
    lr = 1

    state = np.random.rand(side_len**2)
    state = (state > 0.5) * 2 - 1
    state = state.astype(np.short)
    wm = np.zeros(shape=(len(state),len(state)),dtype=np.half)
    
    target_states = np.array([gen_target_state() for i in range(num_ts)]).T
    # for ts in target_states:
    #     for i in range(len(ts)):
    #         for j in range(len(ts)):
    #             if i == j: continue
    #             wm[i,j] += ts[i] * ts[j]
    # W = x_target @ x_target.T
    wm = (target_states@target_states.T).astype(np.half)
    wm = wm*(lr*(1/side_len**2))
    # wm /= len(target_states)
    assert wm.dtype == np.half, wm.dtype
    #update loop
    state_history = []
    energy_history = []
    energy = compute_energy(wm, state, 0)
    energy_history.append(energy)
    state_history.append(state.copy())
    print(energy)

    iterations = 0
    max_iter = 50
    while iterations < max_iter:
        assert state.dtype == np.short
        # for i in range(len(state)):
        #     state[i] = (np.sum([wm[i,j]*state[j] for j in range(len(state))]) > 0) * 2 - 1
        for pixel in np.split(np.random.randint(0,len(state),side_len**2),8):
            #print(wm.shape)
            #print(pixel)
            state[pixel] = np.sign(wm[pixel,:] @ state)
        new_energy = compute_energy(wm, state, 0)
        print(new_energy)
        state_history.append(state.copy())
        energy_history.append(new_energy)
        if energy - new_energy < 1e-4: break
        energy = new_energy
        iterations += 1
    width = max(len(state_history), len(target_states.T))
    fig, ax = plt.subplots(2, width)
    axes = ax.ravel()
    for idx in range(width):
        if idx < len(state_history):
            axes[idx].imshow(np.reshape(state_history[idx], (side_len,side_len)),cmap='binary')
            axes[idx].set_title(f'Timestep {idx}')
    for idx in range(width, width*2):
        if idx - width < len(target_states):
            axes[idx].imshow(np.reshape(target_states.T[idx - width], (side_len, side_len)),cmap='binary')
    plt.show()

if __name__ == "__main__":
    main()
