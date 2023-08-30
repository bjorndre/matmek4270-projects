import numpy as np


def differentiate(u, dt):
    dn = np.zeros(len(u))
    dn[0] = (u[1] - u[0])/(dt)
    dn[-1] = (u[-1] - u[-2])/(dt)
    
    for i in range(1,len(u)-1):
        dn[i] = (u[i+1] - u[i-1])/(2*dt)
    
    return dn

def differentiate_vector(u, dt):
    dn = np.zeros(len(u))
    dn[0] = (u[1] - u[0])/(dt)
    dn[-1] = (u[-1] - u[-2])/(dt)
    dn[1:-1] = (u[2:] - u[:-2])/(2*dt)

    return dn

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()