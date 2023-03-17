import numpy as np
import hopfield as h

def main():
    np.random.seed(0)
    v = np.random.rand(100)
    v = np.array([i > 0.5 for i in v], dtype = np.short)
    

if __name__ == '__main__':
    main()