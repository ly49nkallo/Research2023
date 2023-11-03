import unittest
import numpy as np
from typing import Union, Optional
from collections.abc import Callable
import matplotlib.pyplot as plt

class Hopfield_Continuous(Callable):
    '''Class Defining a Hopfield Energy-Based Associative Memory Network'''
    def __init__(self, 
                 dimension:int, 
                 patterns:Union[np.ndarray, list],
                 beta:Optional[float] = 1) -> None:
        self.dimension = dimension
        self.patterns = patterns if isinstance(patterns, np.ndarray) else np.array(patterns)
        if self.patterns.dtype not in [np.float16, np.float32, np.float64]:
            self.patterns = self.patterns.astype(np.float32)
        # self.beta = beta

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
    
    def __call__(self, state:np.ndarray) -> np.ndarray:
        assert isinstance(state, np.ndarray), f"state must inherit from np.ndarray, got {type(state)}"
        if state.dtype is not np.float32 or state.dtype is not np.float64: state = state.astype(np.float32)
        assert state.ndim == 1 # why does spacial interaction not matter?
        state = state.copy()
        energy = np.infty
        max_iter = 10
        while self.compute_energy(state, self.patterns, beta=1.) < energy - 1e-6 and max_iter > 0:
            state = self.update(state, self.patterns, beta=1)
            max_iter -= 1
        return state
    
class TestHopfild(unittest.TestCase):
    def test1(self):
        '''Remember solid ones'''
        target_state = np.ones((10,10), dtype=np.float16).flatten()
        hf = Hopfield_Continuous(10*10, [target_state])
        init_state = np.random.rand(10,10).flatten()
        out = hf(init_state)
        assert (out == target_state).all() or (out == -target_state).all(), "network did not converge to target state"
