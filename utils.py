import numpy as np
import typing

array_like = typing.Union[np.ndarray, list]

cmap = 'magma'
def stable_softmax(x:array_like):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax