import numpy as np
import hopfield as h
import utils
import matplotlib.pyplot as plt
import matplotlib as m
import scipy.special as ss
from typing import Union, Optional
import warnings

def main():
    target_states = gen_target_states(10)
    state = gen_init_state(type='random')
    
    cntr = 0
    new_energy, energy = 0, 1
    state_history = []
    energy_history = []
    while cntr < 10 and new_energy < energy*0.9:
        state_history.append(state.copy())
        energy = new_energy
        cntr+=1
        state = update(state, target_states, beta=0.25)
        new_energy = compute_energy(state, target_states, beta=0.25)
        energy_history.append(new_energy.reshape((1,)))
        print(new_energy)
    
    display(state_history, target_states = target_states, energy_history=energy_history)
    # plt.imshow(state_history[-2].reshape((h.side_len, h.side_len)))
    # plt.show()
    # print(target_states.shape)

def gen_target_state() -> np.ndarray:
    '''Make a flattened rectangle to memorize'''
    target_state = np.ones((h.side_len,h.side_len), dtype = np.float32)
    target_state[np.random.randint(0, h.side_len//2 - 1): np.random.randint(h.side_len//2 + 1, h.side_len),
                 np.random.randint(0, h.side_len//2 - 1): np.random.randint(h.side_len//2 + 1, h.side_len)]\
                = -np.random.rand()
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
        state = np.random.rand(h.side_len**2) * 2 - 1
        return state
    elif type.lower() == "masked":
        state = gen_target_state()
        state[len(state)//2:] = -1
        return state

def compute_energy(state, target_states, beta=1):
    assert np.max(np.abs(state)) <= 1 + 1e-2, np.max(np.abs(state)) # make sure that the state is within [-1,1]
    M = np.max(np.linalg.norm(target_states, axis = 1))
    energy = -beta**-1 * np.log(np.sum(np.exp(beta*np.dot(target_states, state)))) + 0.5*state@state + beta**-1 * np.log(len(target_states)) + 0.5*M
    return energy


def update(state:np.ndarray, target_states:np.ndarray, beta=1) -> np.ndarray:
    new_state = target_states.T@utils.stable_softmax(beta*np.dot(target_states,state))
    return new_state


def display(state_history:Union[list, np.ndarray], 
            target_states:Union[list, np.ndarray], 
            energy_history:Optional[Union[list, np.ndarray]]=None) -> bool:
    
    if len(target_states) > 30:
        target_states = [target_states[0]]
        warnings.warn("Too many target states to render, defaulting to one")
        
    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(15, 8))
    fig.suptitle("Continuous Hopfield Network")
    gridspec = axs[0].get_subplotspec().get_gridspec()

    subfig1 = fig.add_subfigure(gridspec[0,:])
    subfig2 = fig.add_subfigure(gridspec[1,:])
    subfig3 = fig.add_subfigure(gridspec[2,:])

    subfig1.suptitle('State History')
    subfig1.colorbar(m.cm.ScalarMappable(norm=m.colors.Normalize(-1,1), cmap=utils.cmap))
    axes1 = subfig1.subplots(1, len(state_history))
    for i in range(len(axes1)):
        axes1[i].imshow(state_history[i].reshape((h.side_len, h.side_len)), cmap=utils.cmap, vmin=-1)

    subfig2.suptitle('Target States')
    axes2 = subfig2.subplots(len(target_states)//10 + 1, 10,sharey=True).flatten()
    for i in range(min(len(axes2), len(target_states))):
        axes2[i].imshow(target_states[i].reshape((h.side_len, h.side_len)), cmap=utils.cmap, vmin=-1)

    subfig3.suptitle('Energy History')
    axes3 = subfig3.subplots()
    axes3.plot(energy_history)
    plt.show()
    
    return True

if __name__ == '__main__':
    main()
