import numpy as np
import hopfield as h

def main():
    pass
    input_neurons = -np.ones(10)
    input_neurons[5] = 1;input_neurons[6] = 1;input_neurons[7] = 1;
    hidden_neurons = (np.random.rand(100) > 0.5).astype(np.short)
    print("desired_neurons",desired_output(input_neurons))
    print("input_neurons", input_neurons)
    # matrix from the ith input neuron to the jth hidden neuron


def desired_output(input_neurons:np.ndarray) -> np.ndarray:
    # just flip the input entirely and sum every other together
    # the normalize
    # in retrospect, it is just the AND function but scuffed
    if input_neurons.ndim > 1 or input_neurons.ndim == 0: raise NameError("Input Neurons must have dimension 1")
    if input_neurons.shape[0] % 2 != 0: raise NameError("The number of input neurons must be even")
    out = input_neurons.copy()
    out = -out #flip
    return out.reshape((-1,2)).sum(axis=1) / 2 #sum divide (like AND)

# GLOBAL ENERGY FUNCTION
def energy(wm, state, bias):
    raise NotImplementedError

def set_elem(state,idx,val):
    new_state = state.copy()
    new_state[idx] = val
    return new_state

def update(hidden_state, wm):
    if hidden_state.ndim != 1: raise NameError(f"Hidden state has wrong dimension of {hidden_state.ndims}")
    new_state = np.zeros_like(hidden_state)
    for i in range(hidden_state.shape[0]):
        new_state[i] = np.sign(energy(wm, ...))#@TODO
if __name__ == '__main__':
    main()
