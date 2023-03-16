import numpy as np
import hopfield as h

def main():
    state = np.random.rand(100)
    state = np.array([i > 0.5 for i in state], dtype = np.short)

    
    # assert state.min() == 0 and state.max() == 1, state

if __name__ == '__main__':
    main()