import numpy as np
import hopfield as h
import utils
import matplotlib.pyplot as plt
import scipy.special as ss
from typing import Union, Optional

def main():
    target_states = h.gen_target_states(1000)
    state = h.gen_init_state(type='random')
    
    cntr = 0
    new_energy, energy = 0, 1
    state_history = []
    energy_history = []
    while cntr < 5 and new_energy < energy:
        state_history.append(state.copy())
        energy = new_energy
        cntr+=1
        state = update(state, target_states)
        new_energy = compute_energy(state, target_states)
        energy_history.append(new_energy.reshape((1,)))
        print(new_energy)
    
    display(state_history, target_states = h.gen_target_states(4), energy_history=energy_history)
    # plt.imshow(state_history[-2].reshape((h.side_len, h.side_len)))
    # plt.show()
    # print(target_states.shape)

def compute_energy(state, target_states):
    assert np.max(np.abs(state)) <= 1
    # can spit out -inf if np.dot() returns 0 i.e. if target_states and state are orthoganalx
    # assert False, np.dot(target_states, state)
    # assert np.max(np.exp(np.dot(target_states, state))) != np.infty, np.dot(target_states, state)
    # assert np.sum(np.exp(np.dot(target_states, state))) != np.infty and np.sum(np.exp(np.dot(target_states, state))) > 0, np.sum(np.exp(np.dot(target_states, state)))
    # assert np.log(np.sum(np.exp(np.dot(target_states, state)))) != np.infty, np.log(np.sum(np.exp(np.dot(target_states, state))))
    energy = -np.log(np.sum(np.exp(np.dot(target_states, state), dtype=np.float64)))
    assert energy.size == 1, energy.shape
    assert energy != -np.infty, repr(np.exp(np.dot(target_states, state))) + repr(np.dot(target_states,state))
    return energy

def flip(state, idx):
    new_state = state.copy()
    new_state[idx] = new_state[idx] * -1
    return new_state

def set_elem(state,idx,val):
    new_state = state.copy()
    new_state[idx] = val
    return new_state

def update(state:np.ndarray, target_states:np.ndarray) -> np.ndarray:
    old_state = state.copy()
    for i in range(len(state)):
        #print(-compute_energy(state, target_states) + compute_energy(flip(state, i), target_states))
        try:
            state[i] = np.sign(-compute_energy(set_elem(old_state, i, 1), target_states) + compute_energy(set_elem(old_state, i, -1), target_states))
        except ValueError:
            print(-compute_energy(set_elem(state, i, 1), target_states) + compute_energy(set_elem(state, i, -1), target_states))
            print(compute_energy(set_elem(state, i, 1), target_states), compute_energy(set_elem(state, i, -1), target_states))
            raise ValueError
    return state

def display(state_history:Union[list, np.ndarray], 
            target_states:Union[list, np.ndarray], 
            energy_history:Optional[Union[list, np.ndarray]]=None) -> bool:
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(15, 6))
    gridspec = axs[0].get_subplotspec().get_gridspec()

    subfig1 = fig.add_subfigure(gridspec[0,:])
    subfig2 = fig.add_subfigure(gridspec[1,:])

    subfig1.suptitle('State History')
    axes1 = subfig1.subplots(1, len(state_history))
    for i in range(len(axes1)):
        axes1[i].imshow(state_history[i].reshape((h.side_len, h.side_len)))

    subfig2.suptitle('Target States')
    axes2 = subfig2.subplots(len(target_states)//10 + 1, 10,sharey=True).flatten()
    for i in range(min(len(axes2), len(target_states))):
        axes2[i].imshow(target_states[i].reshape((h.side_len, h.side_len)))
    
    plt.show()

    fig.suptitle('Figure suptitle', fontsize='xx-large')
    plt.show()
    
    return True

if __name__ == '__main__':
    main()
