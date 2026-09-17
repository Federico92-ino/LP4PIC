#Imports
from fbpic.lpa_utils.laser.laser_profiles import LaserProfile
from .profiles_utils import LongitudinalElectricArray,\
                            TransverseElectricArray,\
                            CombinedProfiles
import numpy as np
from scipy.constants import c, epsilon_0, pi
    
class LaserFromDataArrays(LaserProfile):
    def __init__(self,
                 energy,
                 tran_data,
                 x_lims,
                 y_lims,
                 long_data,
                 t_lims,
                 wavelength,
                 pol,
                 propagation_direction=1,
                 gpu_capable=False,
                 z0 = 0.,
                 off_set = [0.,0.],
                 cep = 0.,
                 center_data = False,
                 tran_method = 'linear',
                 long_method = 'linear',
                 filtering = None
                 ):

        tran_profile = TransverseElectricArray(tran_data,
                                               x_lims,
                                               y_lims,
                                               center_data,
                                               off_set,
                                               tran_method,
                                               propagation_direction,
                                               gpu_capable)
        long_profile = LongitudinalElectricArray(wavelength,
                                                 long_data,
                                                 t_lims,
                                                 z0,
                                                 long_method,
                                                 propagation_direction,
                                                 gpu_capable)
        super().__init__(long_profile.propag_direction, long_profile.gpu_capable)
        self.z0 = z0
        self.energy = energy
        dx = tran_profile.dx
        dy = tran_profile.dy
        dt = long_profile.dt
        int_tran = np.trapz(np.trapz(np.abs(tran_data)**2,dx=dy,axis=0),
                          dx=dx) 
        int_long = np.trapz(long_data**2,dx=dt)
        self.__I0__ = energy/(int_tran*int_long)
        self.power = self.__I0__*int_tran
        self.profile = CombinedProfiles(pol,
                                tran_profile,
                                long_profile)
        self.polarization = self.profile.polarization
        self.wavelength = self.profile.long_profile.wavelength
        self.cep = cep
        if filtering is None:
            self.filter = lambda x,y : 1
        else:
            self.filter = filtering

    def E_field(self, x, y, z, t):
        k0 = 2*pi/self.wavelength
        exp_arg = 1j*(k0*(self.propag_direction*z-self.z0-c*t)+self.cep)
        E0 = np.sqrt(2*self.__I0__/c/epsilon_0)
        field = E0*self.profile.evaluate(x,y,z,t)*np.exp(exp_arg)\
                *self.filter(x,y)
        Ex = (field*self.polarization[0]).real
        Ey = (field*self.polarization[1]).real
        return (Ex, Ey)