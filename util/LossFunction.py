import numpy as np
import pytensor.tensor as pt


def squareError(x):
    """Square error loss function."""

    if x.ndim == 1:
        y = pt.vector('y')
        L = pt.mean((x - y) ** 2)

    elif x.ndim == 2:
        y = pt.matrix('y')
        L = pt.mean(pt.sum((x - y) ** 2, axis=1))

    else:
        raise ValueError('x must be either a vector or a matrix.')

    L.name = 'loss'

    return y, L


def crossEntropy(x):
    """Cross entropy loss function. Only works for networks with one output."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x[:, 0]

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = pt.vector('y')
    L = -pt.mean(y * pt.log(x) + (1-y) * pt.log(1-x))
    L.name = 'loss'

    return y, L


def multiCrossEntropy(x):
    """Cross entropy loss function with multiple outputs."""

    assert x.ndim == 2, 'x must be a matrix.'

    y = pt.matrix('y')
    L = -pt.mean(pt.sum(y * pt.log(x), axis=1))
    L.name = 'loss'

    return y, L


def accuracy(x):
    """Accuracy loss function. Mainly useful for validation."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x.argmax(axis=1)

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = pt.vector('y')
    L = 100.0 * pt.mean(pt.eq(y, x))
    L.name = 'loss'

    return y, L


def weightDecay(ws, wdecay):
    """Weight decay regularization."""

    assert wdecay > 0.0

    L = (wdecay / 2.0) * sum([pt.sum(w**2) for w in ws])
    return L


def regularizerSvi(mps, sps, wdecay):
    """
    The type of regularization that is used in stochastic variational inference. Here, we assume that the prior is
    a spherical zero-centred gaussian whose precision corresponds to the weight decay parameter.
    """

    assert wdecay > 0.0

    n_params = sum([mp.get_value().size for mp in mps])

    L1 = 0.5 * wdecay * (sum([pt.sum(mp**2) for mp in mps]) + sum([pt.sum(pt.exp(sp*2)) for sp in sps]))
    L2 = sum([pt.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))

    L = L1 - L2 - Lc

    return L
