import numpy as np
from scipy.special import factorial
from scipy.constants import pi

def __generate_m__(n):
    M = list()
    m = np.arange(1,n+1,dtype=int)
    if n%4==2 or n%4==3:
        if n%2==0:
            M.append(0)
            for i in m[m%2==0]:
                M.extend([-i,i])
        else:
            for i in m[m%2==1]:
                M.extend([-i,i])
    else:
        if n%2==0:
            M.append(0)
            for i in m[m%2==0]:
                M.extend([i,-i])
        else:
            for i in m[m%2==1]:
                M.extend([i,-i])
    return np.array(M)

def cart2pol(x,y):
    r_c = x+1j*y
    r = np.abs(r_c)
    t = np.angle(r_c)
    return r, t

def ZerNorm(m,n):
    if m == 0:
        N = (n+1)/pi
    else:
        N = 2*(n+1)/pi
    return N

def GenerateR(rho,m,n):
    if n >= m:
        pass
    else:
        raise ValueError('n needs to be greater or equal to m')
    R = np.zeros_like(rho)
    if (n-m)%2 == 1:
        R = np.zeros(rho)
    else:
        kmax = int((n-m)/2)
        for k in range(0,kmax+1):
            Prefact = (-1)**k*factorial(n-k)/(factorial(k)*factorial(int((n+m)/2)-k)*factorial(kmax-k))
            n_exp = (n-2*k)
            Rk = Prefact*rho**n_exp
            R += Rk
    return R

def GenerateZernike(rho,phi,m,n):
    if m >= 0:
        Z = GenerateR(rho,m,n)*np.cos(m*phi)
    elif m < 0:
        m = -m
        Z = GenerateR(rho,m,n)*np.sin(m*phi)
    Z = np.where(rho>1,0.,Z)
    return Z

def GenerateZerBank(n,rho,phi):
    ZerBank = dict()
    for N in range(n+1):
        ZerBank[N] = dict()
        m = __generate_m__(N)
        for M in m:
            ZerBank[N][M] = GenerateZernike(rho,phi,M,N)
    return ZerBank