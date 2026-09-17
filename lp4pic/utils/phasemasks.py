import numpy as np
from scipy.constants import pi

def AnnularPhaseMask(Rho,radius,width,N):
    # radius: raggio della regione di spazio su cui fare la mask
    # width: spessore degli anelli/ raggio del cerchio centrale
    # N: numero di anelli di spessore width
    ### N.B: se N==1 fa un cerchio di '1' di raggio=width e zero altrove
    radius -= width
    pm = np.zeros_like(Rho)
    for i in np.linspace(0,1,N):
        pm = np.where((Rho>=i*radius)&(Rho<=i*radius+width),1,pm)
    return pm

def EllipticPhaseMask(X,Y,a,b,width,N):
    if N==0:
        pm = np.zeros_like(X)
        pm = np.where((X**2/a**2+Y**2/b**2)<=1,1,pm)
    else:
        pm = np.zeros_like(X)
        for i in np.linspace(0,N,N):
            pm = np.where((X**2/(a+i*width)**2+Y**2/((b+i*width)**2)>=1)&
                          (X**2/(a+(i+1)*width)**2+Y**2/((b+(i+1)*width)**2)<=1),1,pm)
    return pm

def Pie(T,Ns):
    Ns = int(Ns)
    Theta = T/pi
    theta = np.ones_like(Theta)
    t_slice = np.linspace(-1,1,Ns)
    for i in range(Ns-1):
        theta = np.where((Theta>=t_slice[i])&(Theta<t_slice[i+1]), t_slice[i], theta)
    return theta

def ConcentricPhaseMask(Rho,**kwargs):
    if 'widths' in kwargs:
        widths = kwargs['widths']
        del kwargs['widths']
        N = len(widths)
        values = np.linspace(-1,1+2/(N-1),N+1)
        radii = np.array([0])
        for j in range(1,len(widths)+1):
            radii = np.append(radii,widths[:j].sum())
        radius = 0
    elif 'radius' in kwargs and 'N' in kwargs:
        radius = kwargs['radius']
        N = kwargs['N']
        del kwargs['radius'],kwargs['N']
        values = np.linspace(-1,1+2/(N-1),N+1)
        radii = np.linspace(0,radius,N+1)
    else:
        raise NameError("No 'widths' and/or ('radius','N') were provided")
    pm = np.zeros_like(Rho)
    for j in range(N):
        pm = np.where((Rho>=radii[j])&(Rho<radii[j+1]),
                      values[j]*AnnularPhaseMask(Rho,radius,radii[j+1],1),pm)
    return pm

def SquareMask(X,Y,center,width,height):
    mask = np.ones_like(X)
    mask[(X>=center[0]-width/2)&(X<=center[0]+width/2)&\
         (Y>=center[1]-height/2)&(Y<=center[1]+height/2)] = 0
    return mask