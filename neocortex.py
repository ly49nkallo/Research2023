import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import hopfield_energy_cont as hec

def main():
    # Load MNIST into array
    MNIST_FILEPATH = "mnist_train.csv"
    df = read_csv(MNIST_FILEPATH).to_numpy()[:10000]
    labels = df[:,0]
    data = df[:,1:]
    del df
    # Memorize MNIST dataset into hopfield networks

if __name__ == '__main__':
    main()

