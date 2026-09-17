from fbpic.lpa_utils.laser.longitudinal_laser_profiles import LaserLongitudinalProfile
from fbpic.lpa_utils.laser.transverse_laser_profiles import LaserTransverseProfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.constants import c

def center_of_mass(data):
    img = np.abs(data)
    tmp_img = np.where(img>=img.max()/np.e**.5,img,0)
    r,c = img.shape
    x = np.linspace(0,c-1,c)
    y = np.linspace(0,r-1,r)

    norm = np.sum(tmp_img)
    n0x = np.sum(np.dot(tmp_img,x))/norm
    n0y = np.sum(np.dot(tmp_img.T,y))/norm

    return n0y,n0x

def norm_polarization(pol):
    norm = np.sqrt(np.abs(pol[0])**2+np.abs(pol[1])**2)
    pol[0] = pol[0]/norm
    pol[1] = pol[1]/norm

    return pol

class TransverseElectricArray(LaserTransverseProfile):
    def __init__(self,
                 tran_array,
                 x_lims,
                 y_lims,
                 center_data,
                 offset,
                 method,
                 propagation_direction,
                 gpu_capable):
        super().__init__(propagation_direction, gpu_capable)
        elec_data = tran_array

        ny, nx = np.shape(elec_data)
        self.x = np.linspace(x_lims[0],x_lims[1],nx)
        self.y = np.linspace(y_lims[0],y_lims[1],ny)
        if center_data:
            n_0y,n_0x = np.unravel_index(np.argmax(np.abs(elec_data)**2),elec_data.shape)
            shift = (int(ny/2-n_0y),int(nx/2-n_0x))
            elec_data = np.roll(a=elec_data,
                                shift=shift,
                                axis=(0,1))
            if shift[0] >= 0:
                match np.sign(shift[1]):
                     case 0|1:
                        elec_data = elec_data[shift[0]:,shift[1]:]
                        self.x = self.x[shift[1]:]-self.x[shift[1]:].mean()
                        self.y = self.y[shift[0]:]-self.y[shift[0]:].mean()
                     case -1:
                        elec_data = elec_data[shift[0]:,:shift[1]]
                        self.x = self.x[:shift[1]]-self.x[:shift[1]].mean()
                        self.y = self.y[shift[0]:]-self.y[shift[0]:].mean()
            else:
                match np.sign(shift[1]):
                     case 0|1:
                        elec_data = elec_data[:shift[0],shift[1]:]
                        self.x = self.x[shift[1]:]-self.x[shift[1]:].mean()
                        self.y = self.y[:shift[0]]-self.y[:shift[0]].mean()
                     case -1:
                        elec_data = elec_data[:shift[0],:shift[1]]
                        self.x = self.x[:shift[1]]-self.x[:shift[1]].mean()
                        self.y = self.y[:shift[0]]-self.y[:shift[0]].mean()
        self.set_offset(offset)
        self.dx = self.x[1] -self.x[0]
        self.dy = self.y[1] -self.y[0]
        self.field_interp = RegularGridInterpolator((self.y,self.x),
                                                    elec_data,
                                                    method,
                                                    bounds_error=False,
                                                    fill_value=0.0)
    def set_offset(self,offset):
        self.x = self.x+offset[0]
        self.y = self.y+offset[1]

    def evaluate(self,x,y):
        envelope = self.field_interp((y,x))
        return envelope

class LongitudinalElectricArray(LaserLongitudinalProfile):
    def __init__(self,
                 wavelength,
                 long_array,
                 t_lims,
                 z0,
                 method,
                 propagation_direction,
                 gpu_capable):
        super().__init__(propagation_direction,gpu_capable)
        elec_data = long_array
        nt = len(elec_data)
        self.wavelength = wavelength
        self.t = np.linspace(t_lims[0],t_lims[1],nt)
        self.dt = self.t[1] - self.t[0]
        self.field_interp = interp1d(z0/c+self.t,
                                     elec_data,
                                     method,         
                                     bounds_error=False,
                                     fill_value=0.0)
    def evaluate(self,z,t):
        envelope = self.field_interp(z/c-t)
        return envelope

class CombinedProfiles(object):
    def __init__(self,
                 pol,
                 tran_profile,
                 long_profile):
        self.polarization = norm_polarization(pol)
        self.long_profile = long_profile
        self.tran_profile = tran_profile

    def evaluate(self,x,y,z,t):
        profile = self.tran_profile.evaluate(x,y)*self.long_profile.evaluate(z,t)
        return profile
    