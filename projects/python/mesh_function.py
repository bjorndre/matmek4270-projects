import numpy as np


def mesh_function(f, t):
    mesh_values = np.zeros(len(t))
    for i in range(len(t)):
        mesh_values[i] = f(t[i])
    return mesh_values

def func(t):
    if 0 <= t <= 3:
        return np.exp(-t)
    elif 3 < t <= 4:
        return np.exp(-3*t)
    else:
        raise ValueError("t must be a value between 0 and 4")

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
