from warnings import warn
import numpy as np
from scipy.constants import pi
from scipy.interpolate import RegularGridInterpolator
from numpy.fft import  fftfreq, fftshift
from .ZerUtils import cart2pol, GenerateZerBank
from ..utils.utils import Mirror, __propagator__, color_txt

class __Beam__(object):
    def __init__(self,lambda0,I_dist,x_lims,y_lims):
        if len(I_dist.shape) != 2:
            raise ValueError('I_dist must be a 2darray!')
        self.__ny__,self.__nx__ = I_dist.shape
        self.lambda0 = lambda0
        self.intensity_distribution = I_dist
        self.field_distribution = np.sqrt(I_dist)
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.x = np.linspace(x_lims[0],x_lims[1],self.__nx__)
        self.y = np.linspace(y_lims[0],y_lims[1],self.__ny__)
        self.__dx__ = self.x[1]-self.x[0]
        self.__dy__ = self.y[1]-self.y[0]

class BeamFocusing(__Beam__):
    def __init__(self,Mirror:Mirror,
                 lambda0,I_dist,x_lims,y_lims,
                 norm_radius,oam=0):
        self.Mirror = Mirror
        super().__init__(lambda0,I_dist,
                        x_lims,y_lims)
        # Calculates wavenumbers for the F-transform
        self.__kz__ = 2*pi/self.lambda0
        self.__kx__ = 2*pi*fftshift(fftfreq(self.__nx__,self.__dx__))
        self.__ky__ = 2*pi*fftshift(fftfreq(self.__ny__,self.__dy__))

        self.OrbitalAngularMomentum = oam
        self.transmission = 1.0
        self.__radius__ = norm_radius
        self.__X__,self.__Y__ = np.meshgrid(self.x,self.y)
        self.R,self.T = cart2pol(self.__X__,self.__Y__)

    def __Propagator__(self,f,z):
        return __propagator__(f,self.__dx__,self.__dy__,
                              z,self.__kz__)

    def __reflection__(self,f0,print_out):
        oam = self.OrbitalAngularMomentum
        MirrorShape = self.Mirror.shape

        if not isinstance(oam,int):
            raise ValueError("'oam' must be an integer")
        if print_out:
            print(f"Using a {MirrorShape} mirror "\
                  f"with OAM = {oam} of the beam")
        mirror_phase = self.Mirror.__MirrorPhase__(self.R,self.__kz__)
        f = f0*np.exp(1j*mirror_phase)*np.exp(1j*oam*self.T)
        return f

    def add_aberrations(self, A):
        if not isinstance(A,(list,tuple,np.ndarray)):
            raise ValueError("'A' must be a list, a tuple " \
                             "or a np.ndarray")
        max_n = int((-3+np.sqrt(1+8*len(A)))/2)+1
        MaxNoll = int((max_n+1)*(max_n+2)/2)
        MinNoll = int((max_n)*(max_n+1)/2)
        if len(A) ==  MinNoll:
            pass
        else:
            A = np.append(A,[0.]*(MaxNoll-len(A)))
            warn("The number of aberration coefficients is not compatible\n" 
                 f"with required max radial order {max_n}.\n"
                 f"An aberration array of length = {MaxNoll}"
                 " was filled with zeros for the higher orders.",UserWarning)
        self.Aberrations = A

        W = np.zeros_like(self.R)
        if hasattr(self,'ZernikeBank'):
            if (len(self.ZernikeBank))==max_n or self.ZernikeBank[0][0].shape==self.R.shape:
                pass
            else:
                self.ZernikeBank = GenerateZerBank(max_n,
                                                   self.R/self.__radius__,
                                                   self.T)
        else:
            self.ZernikeBank = GenerateZerBank(max_n,
                                               self.R/self.__radius__,self.T)
        max_n = int((-3+np.sqrt(1+8*len(A)))/2)+1
        for n in range(max_n):
            j_min = int(n*(n+1)/2)
            j_max = int((n+1)*(n+2)/2)
            a = self.Aberrations[j_min:j_max]
            for i,m in enumerate(self.ZernikeBank[n].keys()):
                W += a[i]*self.ZernikeBank[n][m]

        self.ab_field_distribution = \
            self.field_distribution*np.exp(1j*W)
        if hasattr(self,'__focus_cep__'):
            f = self.__reflection__(self.ab_field_distribution,False)
            foo_z = self.__Propagator__(f,self.Mirror.focal_distance)
            i,j = np.unravel_index(np.argmax(np.abs(foo_z)),shape=foo_z.shape)
            self.__focus_cep__ = np.angle(foo_z[i,j])
            del foo_z            

    def propagation(self, z:float|None=None,output='intensity',print_out=False):
        if z is None:
            z = self.Mirror.focal_distance
        elif not isinstance(z,(float,int)):
            raise ValueError('z must be a number!')
        
        if hasattr(self,'ab_field_distribution'):
            f0 = self.ab_field_distribution              
        else:
            f0 = self.field_distribution
        f = self.__reflection__(f0,print_out)

        # Measuring the CarrierEnvelopePhase at the focus the first time
        if  hasattr(self,'__focus_cep__'):
            pass
        else:
            foo_z = self.__Propagator__(f,self.Mirror.focal_distance)
            i,j = np.unravel_index(np.argmax(np.abs(foo_z)),shape=foo_z.shape)
            self.__focus_cep__ = np.angle(foo_z[i,j])
            del foo_z

        #Fourier transform and propagation
        f_z = self.__Propagator__(f,z)
        #Setting the CarrierEnvelopePhase (CEP) of the focus to 0
        f_z = f_z*np.exp(-1j*self.__focus_cep__)

        #Output
        match output:
            case 'complex':
                return f_z
            case 'real':
                return f_z.real
            case 'imag':
                return f_z.imag
            case 'intensity':
                return np.abs(f_z)**2
            case _:
                warn("Chose one in" \
                     "['complex','real','imag','intensity'].\n"\
                     "Default the 'intensity' distribution is returned.")
                return np.abs(f_z)**2

    def apply_phase_mask(self,mask:float|np.ndarray):
        if hasattr(self,'ab_field_distribution'):
            reference = self.ab_field_distribution
            self.ab_field_distribution = self.ab_field_distribution*np.exp(1j*mask)
            self.field_distribution = self.field_distribution*np.exp(1j*mask)
            self.transmission = self.transmission*np.sum(np.abs(self.ab_field_distribution)**2)/np.sum(np.abs(reference)**2)
        else:
            reference = self.field_distribution
            self.field_distribution = self.field_distribution*np.exp(1j*mask)
            self.transmission = self.transmission*np.sum(np.abs(self.field_distribution)**2)/np.sum(np.abs(reference)**2)

    def substitute_intensity(self,new_dist:np.ndarray):
        self.__init__(self.Mirror,self.lambda0,new_dist,
                      self.x_lims,self.y_lims,self.__radius__,self.OrbitalAngularMomentum)
        print(color_txt(f"Substituting the intesity distribution\n"\
                        "will delete any phase mask already initialized.",255,255,0),
                          flush=True)
        if hasattr(self,'Aberrations'):
            self.add_aberrations(self.Aberrations)