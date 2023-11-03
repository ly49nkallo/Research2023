import numpy as np
from pandas import read_csv
from typing import Union, Optional, Any
from collections.abc import Callable

def main():
    # Load MNIST into array
    MNIST_FILEPATH = "mnist_train.csv"
    df = read_csv(MNIST_FILEPATH).to_numpy()[:10000]
    labels = df[:,0]
    data = df[:,1:]
    del df
    # Memorize MNIST dataset into hopfield networks
    networks = []
    for i in range(100):
        networks.append(Hopfield_Continuous(28**2, [data[i:i+10]]))

class Hopfield_Continuous(Callable):
    '''Class Defining a Hopfield Energy-Based Associative Memory Network'''
    def __init__(self, 
                 dimension:int, 
                 patterns:Union[np.ndarray, list],
                 beta:Optional[float]):
        self.dimension = dimension
        self.patterns = patterns if isinstance(patterns, np.ndarray) else np.array(patterns)
        self.beta = beta

    @property
    def size(self): return self.dimension    
    def __len__(self): return self.size
    @property
    def beta(self): return self.beta
    
    @staticmethod
    def compute_energy(state, target_states, beta=1):
        assert np.max(np.abs(state)) <= 1 + 1e-2, np.max(np.abs(state)) # make sure that the state is within [-1,1]
        M = np.max(np.linalg.norm(target_states, axis = 1))
        energy = -beta**-1 * np.log(np.sum(np.exp(beta*np.dot(target_states, state)))) + \
        0.5*state@state + beta**-1 * np.log(len(target_states)) + 0.5*M
        return energy
    
    @staticmethod
    def update(state:np.ndarray, target_states:np.ndarray, beta=1) -> np.ndarray:
        def stable_softmax(x:np.ndarray):
            z = x - max(x)
            numerator = np.exp(z)
            denominator = np.sum(numerator)
            softmax = numerator/denominator
            return softmax
        
        new_state = target_states.T@stable_softmax(beta*np.dot(target_states,state))
        return new_state
    
    def __call__(self, state:np.ndarray):
        assert isinstance(state, np.ndarray), f"state must inherit from np.ndarray, got {type(state)}"
        if state.dtype is not np.float32 or state.dtype is not np.float64: state = state.astype(np.float64)
        assert state.ndim == 1 # why does spacial interaction not matter?
        state = state.copy()
        energy = np.infty
        max_iter = 10
        while self.compute_energy(state, self.patterns, beta=1.) < energy - 1e-6 and max_iter > 0:
            state = self.update(state, self.patterns, beta=1)
            max_iter -= 1
        return state
    
if __name__ == '__main__':
    main()

