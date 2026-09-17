"""
This script is an example FBPIC simulation in LorentzBoost frame, with laser initialization from a LP4PIC.
It works in two modalities, with a syntethic beam profile or with beam reconstruction from fluence measurements 
"""
# -------
# Imports
# -------
import numpy as np
from scipy.constants import c, e, m_e, pi, epsilon_0, m_p
r_e = e**2/(4*pi*epsilon_0*m_e*c**2) # classical electron radius
from skimage.filters import gaussian
import json
import os
import copy
import matplotlib.pyplot as plt
# Imports from FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic
# Import from LP4PIC
from lp4pic import ReadImages, Mirror,BeamFocusing, LaserFromDataArrays, PhaseReconstructor, GeneticOptimizer
from lp4pic.utils.phasemasks import Pie
from lp4pic.utils import color_txt, WorkingPlaneTransform
from lp4pic.focusing.ZerUtils import cart2pol
# Custom functions
def w(z):
   return np.sqrt(1+z**2)
def SuperGauss(r,z,lambda0,w0,order=1):
    kz = 2*pi/lambda0
    zr = kz*w0**2/2
    W = w0*w(z/zr)
    r_n = r/W
    field_dist = w0/W*np.exp(-r_n**(2*order))
    return field_dist

#%% ----------
# Parameters
# ----------
use_cuda = True
n_order = 64
"""
This script works with two InputMod options. Choose the InputMod among:
- 'ExpNF' : Experimental NearField measurement as input (check the 'source' folder for meta-data in the relative section)
- 'ArtSG': an artificial Super-Gaussian profile as input (check the parameters in the relative 'case'-section of the following 'match')
- 'Recon': With this InputMod the BeamFocusing object to retrieve fields is obtained from the full reconstruction procedure with Near- and FarField iamges.
"""
# PRELIMINARY: LP4PIC beam focusing 
# General input params for LP4PIC 
lambda0 = 810e-9
oam = 0
mirror_shape = 'Parabolic' # Can be also a function f(R,kz,F)
apply_phase_mask = False # Simulate the passing through a phase mask generating a LaguerreGauss_01 beam in the focus
Aberrations = None     ## if a list/tuple it activates the .add_aberrations
                       ## with the given coefficients; just for 'ArtSG' and 'ExpNF' Inputmod
InputMod = 'Recon'
saving_focal_spot_image = None ## if a path (str) is given,
                               ## this will create an image named
                               ## '/path/focal_spot.png'

match InputMod:
    case 'ArtSG':
        #NearField params:
        SG_order = 3
        w_nf = 6e-2
        d_nf = pi*w_nf
        #FarField params:
        w_ff = 22e-6
        F_dist = w_ff*d_nf/lambda0
        FN = F_dist/d_nf
        ## Box params to generate SG intensity profile (w/o OAM)
        Nx = 2000
        x_lims,y_lims = [np.array([-3*w_nf,3*w_nf]) for i in range(2)]
        x_nf, y_nf = [np.linspace(lims[0],lims[1],Nx) for lims in [x_lims,y_lims]]
        X,Y = np.meshgrid(x_nf,y_nf)
        Rho,T = cart2pol(X,Y)
        ## hand-made WP transform
        Nw = 30                       # |  A Nw = int(w_ff/dx_w) -theoretical FF waist resolution in pixels
        dx = x_nf[1]-x_nf[0]          # |  | dx_w = s*dx
        dx_w = w_ff/Nw                # |  | dx = x_nf[1]-x_nf[0]
        s = dx_w/dx                   # |  | s = F/F_dist
        F = F_dist*s                  # |  | F = NzR*zR
        zR = pi*w_ff**2/lambda0       # |  | zR = pi*w_ff**2/lambda0
        NzR = int(F/zR)               # V  | NzR = 5      -working plane distance in Rayleigh lengths
        ## BeamFocusing object initialization
        norm_r = d_nf/2*s
        NF_profile = np.abs(SuperGauss(Rho,0,lambda0,w_nf,order=SG_order))**2/s**2
        print(color_txt("Using artificial Super-Gaussian profile as input\n"+\
                        f"Collimated beam with diameter of {d_nf*1e3:.0f} mm, with parabola of F = {F_dist*1e2:.0f} cm, {FN = :.0f}\n"+\
                        f"Working plane at {NzR} Rayleigh lengths from focus\n",
                        255,128,0),flush=True)
        #Choose the mirror
        mirror = Mirror(F_dist=F,shape=mirror_shape)
        #Generate BeamFocusing instances
        Beam = BeamFocusing(Mirror=mirror,
                            lambda0=lambda0,
                            I_dist=NF_profile,
                            x_lims=x_lims*s,
                            y_lims=y_lims*s,
                            norm_radius=norm_r,
                            oam=oam)
    case 'ExpNF':
        #The laser "work" params: input images loading
        source = "./imgs/source"  # here goes the folder where a file 'images_params.json' is located
        with open(f'{source}/images_params.json', 'r') as fp:
            img_params = json.load(fp)
        """
        README for ReadImages params:
            N.B: We need the "images_params.json" file to contain at least the following fields:
            - F_dist: focal length of the parabolic mirror (in meters).
            - NFpath: path to the NearField image.
            - calibNF: calibration of the NearField image (in meters/pixel).
        - resXX: factor for changing the image resolution. If resXX < 1 the image is downsampled, if resXX > 1 the image is upsampled.
        - threshXX: threshold for contour detection; now in [0,1] representing the percentage of the max pixel value.

        - XXcom_mode: method for centering the image according to contours detection. Choose among:
                    *'imageCOM' : center of mass of the image.
                    *'parent'   : center of mass of the biggest contour found (1st level).
                    *'son'      : if any, center of mass of the biggest contour found inside the biggest parent (2nd level).
                    *'grandson' : if any, center of mass of the biggest contour found inside the biggest son (3rd level).

        - XXremove_bg: method for removing background noise. Choose between:
                    *'masking'   : zeroing pixels outside the biggest detected contour at 'parent' level (very sharp edges).
                    *'filtering' : applying a SuperGaussian filter over the biggest contour detected at 'parent' level (soft edges).
                                   Remember to set the order of the SG filter through 'XXfilt_order' variable.
        - XXblur_dict: dictionary containing the parameters for a Gaussian blur pre-processing of the image.
                    *'blur_it' : number of times the blur is applied.
                    *'ksize'   : kernel size for the Gaussian blur.
                    *'sigma'   : standard deviation for Gaussian kernel.
        """
        F_dist = img_params["F_dist"]
        NFpath = img_params["NFpath"]
        calibNF = img_params["calibNF"]
        resNF = 1
        threshNF = 0.1
        offsetNF={'x':0,"y":0}
        NFcom_method = 'parent'
        NFremove_bg = 'filtering'
        NFfilt_order = 20
        NFblur_dict = {'blur_it':1,
                        'ksize':(15,15),
                        'sigma':0}
        ##Reading images
        ImgNF = ReadImages(NFpath,
                          calibration=calibNF,
                          threshold=threshNF,
                          com_mode=NFcom_method,
                          remove_bg=NFremove_bg,
                          SGfilt_order=NFfilt_order,
                          blur_dict=NFblur_dict,
                          change_res=resNF,
                          square_img=True,
                          offset=offsetNF,
                          activate_blur=True) # after the image preprocessing, with this keyword ReadImages return a blurred profile or not.
        ## Params of the experimental NF
        x_lims,y_lims = [np.array(ImgNF.limits[i]) for i in ['x','y']]
        x_nf, y_nf = [ImgNF.x, ImgNF.y]
        d_nf = ImgNF.estimated_beam_diameter
        w_nf = d_nf/2
        X,Y = np.meshgrid(x_nf,y_nf)
        Rho,T = cart2pol(X,Y)
        #Generate BeamFocusing instances
        NzR = 5
        Beam = BeamFocusing(*WorkingPlaneTransform(ImgNF,
                                                   None,
                                                   Mirror(F_dist,mirror_shape),
                                                   lambda0,
                                                   NzR),
                            oam=oam)
        s = (Beam.x[1]-Beam.x[0])/(x_nf[1]-x_nf[0])
    case 'Recon':
        source = f"./imgs/source"
        with open(source+'/images_params.json','r') as fp:
            img_params = json.load(fp)
            F_dist = img_params['F_dist']
            NFpath = img_params['NFpath']
            calibNF = img_params['calibNF']
            calibFF = img_params['calibFF']
            planes_pos = img_params['planes_position']
            focus_index = np.where(np.array(planes_pos)==0.0)[0][0]
            FFpaths = img_params['FFpath'][focus_index:focus_index+1]
            planes_pos = img_params['planes_position'][focus_index:focus_index+1]

        ##Reading images
        threshNF = 0.1
        NFcom_method = 'son'  # str: 'imageCOM'|'parent'|'son'|'grandson'|'max'
        NFremove_bg = 'masking' #str| None: 'masking'|'filtering'| None 
        NFfilt_order = 20
        NFblur_dict={'blur_it':1,
                     'ksize':(15,15),
                     'sigma':0}
        resNF = 1
        offsetNF={'x':0,"y":0}
        ImgNF = ReadImages(NFpath,
                          calibration=calibNF,
                          threshold=threshNF,
                          com_mode=NFcom_method,
                          remove_bg=NFremove_bg,
                          mask_bg=None,
                          SGfilt_order=NFfilt_order,
                          blur_dict=NFblur_dict,
                          change_res=resNF,
                          square_img=True,
                          offset=offsetNF,
                          activate_blur=True)
        threshFF = [0.1]
        FFcom_method = 'parent'
        FFremove_bg = None
        FFfilt_order = 20
        FFblur_dict={'blur_it':1,
                     'ksize':(75,75),
                     'sigma': 0}
        resFF = 1
        offsetFF={"x":0,"y":0}
        ImgsFF = [ReadImages(FFpath,
                          calibration=calibFF,
                          normalize=False,
                          threshold=threshFF[j],
                          com_mode=FFcom_method,
                          remove_bg=FFremove_bg,
                          mask_bg = None,
                          SGfilt_order=FFfilt_order,
                          blur_dict=FFblur_dict,
                          change_res=resFF,
                          center_image=True,
                          square_img=True,
                          offset=offsetFF,
                          activate_blur=True) for j,FFpath in enumerate(FFpaths)]
        planes = img_params["planes_position"]
        focus_index = np.where(np.array(planes_pos)==0.0)[0][0]
        for ImgFF in ImgsFF:
            ImgFF.renormalize(ImgNF.image_charge)
        ############################################
        ### __Gerchberg-Saxton phase retrieval__ ###
        ############################################
        #Inputs to build the Phase Reconstructor
        #[0] for single image or img_params['planes_position'] if it is in .json file
        mirror = Mirror(F_dist,'Parabolic')
        lambda0 = .81e-6
        seed = 0
        ## Inputs for methods in the Phase Reconstructor:
        #1) Phase retrieval
        NzR = 10
        N_gs = 100
        focus_index = np.where(np.array(planes_pos)==0.0)[0][0]
        adjust_F = False
        mod = 'Sequential'
        mod_dim = 11
        #2) Phase projection
        max_n_projection = 10 
        phase_ret = PhaseReconstructor(ImgNF, ImgsFF, mirror, lambda0,
                                       planes_pos=planes_pos,seed_phase=seed,save_dir=None)
        phase_ret.GerSaxPhaseRetriever(NzR,N_gs,focus_index,adjust_F,align_max=False,
                                       mod=mod,mod_dim=mod_dim,loss_func='MSE')
        Error = phase_ret.error/phase_ret.error[0]*100
        # Phase projection: building of Aberrated (Zernike) Beam
        Aberrations, projected_phase = phase_ret.Aberrometer(max_n_projection,
                                                             'unwrapped')
        #############################################
        ### __Genetic Optimization of projected__ ###
        #############################################
        #Inputs for the optimizer
        Nevolutions = 70
        Ntrials = 100
        BestError=1e30
        NzR=10

        "These three the searching space for mutations"
        ScanAmplitude_max = 1.2
        ScanAmplitude_min = .01
        DecayIteration = 8

        limits = {'x':[-250e-6,250e-6],'y':[-250e-6,250e-6]} #This variable limits the ROI at focal plane in checking the error 
        # Genetic Optimization --> 2nd step to optimize the ab coefficients
        ##Re-init ImgNF with lower resolution to speed up
        """
        WARNING: We have noticed that for the GeneticOptimisation one desn't need the full resolved NearField image.
         It works also with an undersolved image, that is why we re.read a temporary NF image, then we substitute to the obtained
         BeamFocusing object the original, full-resolved profile.
        """
        tmp_ImgNF = ReadImages(NFpath,
                          calibration=calibNF,
                          threshold=threshNF,
                          com_mode=NFcom_method,
                          remove_bg=NFremove_bg,
                          SGfilt_order=NFfilt_order,
                          blur_dict=NFblur_dict,
                          change_res=.3,
                          square_img=True,
                          offset=offsetNF,
                          activate_blur=True)

        Beam, Iterations, Errors = GeneticOptimizer(tmp_ImgNF,ImgFF,
                                                    mirror,lambda0,NzR,
                                                    Nevolutions,Ntrials,
                                                    best_err=BestError,
                                                    scan_amp_max=ScanAmplitude_max,
                                                    scan_amp_min=ScanAmplitude_min,
                                                    decay_it=DecayIteration,
                                                    ab_array=Aberrations,
                                                    save_dir=None,
                                                    ab_rnd_search=False,
                                                    save_rnd=False,
                                                    save_evo=False,
                                                    adjust_F=adjust_F,
                                                    loss_func='MSE',
                                                    limits=limits)

        Aberrations_opt = Beam.Aberrations
        s = Beam.x_lims[0]/ImgNF.limits['x'][0]
        Beam.substitute_intensity(ImgNF.image/s**2)

# Applying phase mask and/or given aberrations   
if apply_phase_mask:
    r_dim_mask = d_nf/2
    hole_ratio = .5
    phase_mask = np.ones_like(Rho)
    phase_mask[Rho>r_dim_mask]=0
    phase_mask = phase_mask*Pie(T,20)
    Phase = pi*phase_mask
    Beam.apply_phase_mask(Phase)
if InputMod in ['ArtSG','ExpNF']:
    if Aberrations is not None:
        print(color_txt("Adding aberrations to the beam...",0,128,255),flush=True)
        n_max_Zer = int(.5*(np.sqrt(8*len(Aberrations)+1)-3))
        Beam.add_aberrations(Aberrations)
else:
    pass

if saving_focal_spot_image is not None:
    extentNF = np.array([x_lims[0],x_lims[1],y_lims[0],y_lims[1]])*1e2
    extentFF = extentNF*s*1e4
    origin='lower'
    aspect='equal'
    cut_nf = 2e2*w_nf
    cut_ff = 10e6*w_ff
    cmap_intensity='afmhot'
    if apply_phase_mask:
        mosaic = [['A','B','C']]
        fig0,ax = plt.subplot_mosaic(mosaic,
                                 figsize=(13.5,4.5),
                                 gridspec_kw={'wspace':0,'hspace':0})
        title = f"Phase mask"
        cmap='Greys_r'
        vmin=-pi
        vmax=pi
        cb_ticks = [0,pi]
        cb_ticks_labels = ['0',r'$\pi$']
        label_cb = '(rad)'
        phase=ax['B'].imshow(Phase,
                     cmap=cmap,
                     extent=extentNF,
                     aspect=aspect,
                     origin=origin,
                     vmin=vmin,
                     vmax=vmax)
        ax['B'].set_xlabel(r'x',size=12)
        ax['B'].set_ylabel(r'y',size=12,rotation='horizontal')
        ax['B'].tick_params(labelsize=12)
        ax['B'].set_xlim(-cut_nf,cut_nf)
        ax['B'].set_ylim(-cut_nf,cut_nf)
        ax['B'].set_title(title,size=14)
        cb_fase=fig0.colorbar(phase,
                     orientation='horizontal',
                     shrink=0.5,
                     pad=.15,)
        cb_fase.set_label(label_cb)
        cb_fase.set_ticks(cb_ticks)
        cb_fase.set_ticklabels(cb_ticks_labels)
    else:
        mosaic = "AC"
        fig0,ax = plt.subplot_mosaic(mosaic,
                                     figsize=(13.5,4.5),
                                     gridspec_kw={'wspace':0,'hspace':0})
    NF_IntensityMap = Beam.intensity_distribution
    imgnf=ax['A'].imshow(NF_IntensityMap*s**2,
                     cmap=cmap_intensity,
                     extent=extentNF,
                     aspect=aspect,
                     origin=origin,
                     vmin=0,vmax=1)
    ax['A'].set_xlabel(r'x (cm)',size=12)
    ax['A'].set_ylabel(r'y (cm)',size=12)
    ax['A'].set_xlim(-cut_nf,cut_nf)
    ax['A'].set_ylim(-cut_nf,cut_nf)
    ax['A'].tick_params(labelsize=12)
    ax['A'].set_title(f"NearField",size=14)
    fig0.colorbar(imgnf,
                 orientation='horizontal',
                 shrink=.5,
                 pad=.17).set_label('intensity (a.u.)')

    FF_IntensityMap = Beam.propagation(output='intensity')
    imgff=ax['C'].imshow(FF_IntensityMap,                              
                     cmap=cmap_intensity,
                     origin=origin,
                     extent=extentFF,
                     aspect=aspect,
                     norm=plt.matplotlib.colors.PowerNorm(gamma=0.5,vmin=0))
    ax['C'].set_xlabel(r'x ($\mu$m)',size=12)
    ax['C'].set_ylabel(r'y ($\mu$m)',size=12,labelpad=.07)
    ax['C'].tick_params(labelsize=12)
    ax['C'].set_title('Focal spot')
    ax['C'].set_xlim(-cut_ff,cut_ff)
    ax['C'].set_ylim(-cut_ff,cut_ff)
    cb_ff=fig0.colorbar(imgff,
                 orientation='horizontal',
                 shrink=0.5,
                 pad=.17)
    cb_ff.set_label('intensity (a.u.)')
    cb_ff.ax.ticklabel_format(scilimits=(1,3))
    fig0.savefig(f"{saving_focal_spot_image}/focal_spot.png")

#%% FBPIC SIMULATION: preliminaries
# General simulation parameters
ActivateIonizationDopant = True
######PLASMA######
#The electron plasma and related quantities
n_e = 10e17*1.e6                           # The density in the labframe (electrons.meters^-3)
omegap = np.sqrt(e**2*n_e/(m_e*epsilon_0)) # Plasma frequency
lambdap = 2*pi*c/omegap                    # Plasma linear wavelength
gammap = (2*pi*c/lambda0)/omegap           # Plasma gamma
betap = np.sqrt(1-1/gammap**2)             # Plasma wave phase velocity
print(f"Plasma wavelength = {lambdap*1e6} um")
#Target constituents
if ActivateIonizationDopant:
	ContFraction=0.2
	ContLength = 0.3e-3
	Contaminant = 'N'
	ID = 5 
	n_He = n_e/2/(1+ID*ContFraction/(1-ContFraction))
	n_N2 = n_He*ContFraction/(1-ContFraction)
	n_N  = 2*n_N2
	print (f"Target densities for electrons, Helium and atomic contaminants are {n_e*1e-6:.2e}, {n_He*1e-6:.2e}, {n_N*1e-6:.2e} cm^-3 respectively")
	print (f"Total charge density is {((2*n_He+ID*n_N)-n_e)*1e-6:.2e} cm^-3")
else:
    ContLength = 0.
    n_He=n_e/2
    n_N2=0
#Density profile: define  density function
ramp_up = 0.2e-3
plateau = 1e-3
ramp_down = 0.2e-3
def dens_func( z, r ):
    # Allocate relative density
    n = np.ones_like(z)
    # Make ramp up
    inv_ramp_up = 1./ramp_up
    n = np.where( z<=ramp_up, np.sin(.5*pi*z*inv_ramp_up)**2, n )
    n = np.where(z<0,0,n)
    # Make ramp down
    inv_ramp_down = 1./ramp_down
    n = np.where( (z > ramp_up+plateau) & \
                  (z <= ramp_up+plateau+ramp_down),
              np.cos(.5*pi*(z-(ramp_up+plateau))*inv_ramp_down)**2, n )
    n = np.where( z > ramp_up+plateau+ramp_down, 0, n)
    return(n)

######LASERS#####
#Driver laser pulses (lp4pic: FBPIC interface)
def filter_out(x,y):
    """
    Filter function to remove laser pulse residual 
    at box boundaries.
    """
    f = np.exp(-1.5*((x**2+y**2)/rmax**2)**8)
    return f
omega0 = 2*pi*c/lambda0            # Laser central frequency
n_c = m_e*epsilon_0*omega0**2/e**2 # critical density (m^-3)

# The laser object
tau_FWHM = 25e-15 	  # Laser FWHM duration (s)
tau = tau_FWHM/np.sqrt(np.log(4))
t = np.linspace(-3*tau,3*tau,2000)
ctau = c*tau
z0 = -3*ctau          # front pulse centroid (m)
FocalDistance = Beam.Mirror.focal_distance
FocalPosition = 1200e-6
# temporal profile
Ez = np.exp(-(t**2/tau**2))
# transverse distribution
Er  = Beam.propagation(output='complex',z=FocalDistance-(FocalPosition-z0))
kernel = 1 # higher odd integer to apply gaussian filtering if needed
sigma = .3*(.5*(kernel-1)-1)+.8
truncate = int(.5*(kernel-1)/sigma-.5)
Er = gaussian(Er.real,sigma=sigma,truncate=truncate)+1j*\
      gaussian(Er.imag,sigma=sigma,truncate=truncate)

PEC_calculation = 'given_a0' # Pulse Energy Content calculation method
match PEC_calculation:
    case 'given_a0':
        # Searching for normalization factor
        tmp_Focus = Beam.propagation() # needed to calculate the norm_value which provide
                                        # the wanted a0 in the hole@focus position according to the peak
        norm_value = np.max(tmp_Focus)
        a0 = 2
        i0 = c/(8*epsilon_0)*(e/r_e)**2*a0**2/lambda0**2
        total_energy = i0*np.trapz(np.trapz(abs(Er)**2,x=Beam.x),x=Beam.y)/norm_value\
                    *np.trapz(abs(Ez)**2,t) # Pulse energy (J)
    case 'given_energy':
        total_energy = 20 # Pulse energy (J)
        norm_value = 1 # not used in this case
pol1 = np.array([1,0])
print(color_txt("Initializing laser objects",255,255,255),flush=True)
print(color_txt("Filtering the pulse....",255,0,0),flush=True)
laser_profile = LaserFromDataArrays(energy=Beam.transmission*total_energy,
                                  tran_data=Er/np.sqrt(norm_value), x_lims=Beam.x_lims, y_lims=Beam.y_lims,
                                  long_data=Ez, t_lims=[t.min(),t.max()], z0=z0,
                                  wavelength=lambda0, pol=pol1, propagation_direction=1,
                                  filtering=filter_out,center_data=False)
print(color_txt(".... done",0,255,0),flush=True)
#####The simulation init#####
##Boosted frame
gamma_boost = .3*gammap
boost = BoostConverter(gamma_boost)
print("Lorentz boosted frame simulation with gamma = %g" %gamma_boost)
##The simulation box
ppl_z = 40                             # PointPerLambda resolution in z
ppl_r = 8                              # PointPerLambda resolution in r
dz = lambda0/ppl_z                     # Longitudinal resolution (meters)
dr = lambda0/ppl_r	                   # Transverse resolution (meters)
zmax = 0                               # Length of the box along z (meters)
zmin = z0-lambdap-3*ctau
rmax = 12*(1.2*w_ff)                    # Length of the box along r (meters)
window_length = zmax-zmin
Nz = int((zmax-zmin)/dz)               # Number of gridpoints along z
Nr = int(rmax/dr)                      # Number of gridpoints along r
Nm = 2                                 # Number of modes used
##The particles of the plasma
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = p_zmin+ramp_up + plateau + ramp_down
p_rmax = 0.95*rmax # Maximal radial position of the plasma (meters)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4*Nm      # Number of particles per cell along theta
##The simulation timestep
dt = min( rmax/(2*boost.gamma0*Nr)/c, (zmax-zmin)/Nz/c )  # Timestep (seconds)
##The moving window (moves with the group velocity in a plasma)
v_window = c*betap
#Velocity of the Galilean frame (for suppression of the NCI)
v_comoving= - c * np.sqrt( 1. - 1./boost.gamma0**2 )
##The interaction length of the simulation, in the lab frame (meters)
L_interact = p_zmax+window_length+100e-6 
# Interaction time, in the boosted frame (seconds)
T_interact = boost.interaction_time( L_interact, (zmax-zmin), v_window )
# (i.e. the time it takes for the moving window to slide across the plasma)

#######The diagnostics init######
# Number of discrete diagnostic snapshots, for the diagnostics in the
# boosted frame (i.e. simulation frame) and in the lab frame
# (i.e. back-transformed from the simulation frame to the lab frame)
N_lab_diag = 20+1
# Time interval between diagnostic snapshots *in the lab frame*
# (first at t=0, last at t=T_interact)
dt_lab_diag_period = (L_interact + (zmax-zmin)) / v_window / (N_lab_diag - 1)
# Period of writing the cached, backtransformed lab frame diagnostics to disk
write_period = 400
# Whether to tag and track the particles of the bunch
track_ptcl = True
sub_frac = 1

#######Saving simulation parameters######
# Directory for saving simulation outputs
write_dir =  f'./{InputMod}'
os.makedirs(write_dir,exist_ok=True)

activate_saving_params = False # if 'True' it saves in .json format the following parameters
if activate_saving_params:
    print(color_txt("Saving simulation parameters to json file...",0,255,0),flush=True)
    params = dict()
    params['Nr'] = Nr
    params['Nz'] = Nz
    params['zmin'] = zmin
    params['n_e'] = n_e
    params['omegap'] = omegap
    params['omega0'] = 2*pi*c/lambda0
    params['v_window'] = v_window
    params['subsampling_fraction'] = sub_frac
    params['simulation'] = {'resolution':{'dz': dz,
                                          'dr' : dr,
                                          'dt': dt, 
                                          'ppl':{'z': ppl_z, 'r': ppl_r}},
                            'ptcl_per_cell':{'tot':p_nz*p_nr*p_nt,
                                             'z':p_nz,
                                             'r':p_nr,
                                             'tetha':p_nt},
                            'grid':{'Nz':Nz,'Nr':Nr,'Nm':Nm},
                            'box_size':{'zmin':zmin,'zmax':zmax,'rmax':rmax},
                            'boosted_frame':{'gamma_boost':boost.gamma0,
                                             'galileian_v_comoving':v_comoving}
                            }
    params['plasma'] = {'elec_bg_density':f"{n_e*1e-6:.2e}cm^-3",
                        'nozzle':{'start':p_zmin,
                                  'up_ramp':ramp_up,
                                  'plateau':plateau,
                                  'downramp':ramp_down,
                                  'contaminant_length':{'from':p_zmin,'to':p_zmin+ContLength}},
                        'gamma_p':gammap,
                        'lambda_p':lambdap,
                        'omega_p':omegap,
                        'gas_mixture':{'He':f"{n_He*1e-6:.2e}cm^-3,{n_He/(n_He+n_N2)*100:.1f}%",
                                       'N2':f"{n_N2*1e-6:.2e}cm^-3,{n_N2/(n_He+n_N2)*100:.1f}%",
                                       'contaminant_ionization_level':ID}}
    params['laser']={'energy':laser_profile.energy,
                     'power':laser_profile.power,
                     'polarization':str(laser_profile.polarization.tolist()),
                     'lambda0':laser_profile.wavelength,
                     'duration':tau,
                     'waist':w_ff,
                     'centroid':laser_profile.z0}
    if PEC_calculation=='given_a0':
        params['laser']['a0'] = a0
        params['laser']['intensity'] = laser_profile.__I0__
        params['laser']['check_i0_hole'] = i0

    with open(write_dir+'/params.json','w') as fp:
        json.dump(params,fp,indent=4)
# ---------------------------
# Carrying out the simulation
# ---------------------------
# NB: The code below is only executed when running the script,
# (`python boosted_frame_sim.py`), but not when importing it.
if __name__ == '__main__':
    
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
                      v_comoving=v_comoving, gamma_boost=boost.gamma0,
                      n_order=n_order, use_cuda=use_cuda,
                      boundaries={'z':'open', 'r':'open'})
                      # 'r': 'open' can also be used, but is more computationally expensive
    # Add the plasma electron and plasma ions
    plasma_elec_Acc = sim.add_new_species( q=-e, m=m_e, n=n_e,
                    dens_func=dens_func, boost_positions_in_dens_func=True,
                    p_zmin=p_zmin+ContLength, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
    He_ions_Acc = sim.add_new_species( q=2*e, m=4*m_p, n=n_e/2,
                    dens_func=dens_func, boost_positions_in_dens_func=True,
                    p_zmin=p_zmin+ContLength, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
    if ActivateIonizationDopant:
        plasma_elec_Inj = sim.add_new_species( q=-e, m=m_e, n=n_e,
                        dens_func=dens_func, boost_positions_in_dens_func=True,
                        p_zmin=p_zmin, p_zmax=p_zmin+ContLength, p_rmax=p_rmax,
                        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
        He_ions_Inj = sim.add_new_species( q=2*e, m=4*m_p, n=n_He,
                        dens_func=dens_func, boost_positions_in_dens_func=True,
                        p_zmin=p_zmin, p_zmax=p_zmin+ContLength, p_rmax=p_rmax,
                        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
        elec_from_N = sim.add_new_species( q=-e, m=m_e )
        atoms_N = sim.add_new_species( q=ID*e, m=14.*m_p, n=n_N,
        	            dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin,
                                       p_zmax=ContLength, p_rmax=p_rmax, boost_positions_in_dens_func=True )
        atoms_N.make_ionizable('N', target_species=elec_from_N, level_start=ID )
        if track_ptcl:
            elec_from_N.track(sim.comm)
    # Adding pulses
    # Add the driver pulses
    add_laser_pulse( sim, laser_profile, gamma_boost=boost.gamma0,
                     method='antenna',z0_antenna=zmin+5e-6)
        
    # Convert parameter to boosted frame
    v_window_boosted, = boost.velocity( [ v_window ] )
    # Configure the moving window
    sim.set_moving_window( v=v_window_boosted )
    # Add a field diagnostic
    sim.diags = [# Diagnostics in the lab frame (back-transformed)
                  BackTransformedFieldDiagnostic( zmin, zmax, v_window, dt_lab_diag_period,
                                                  N_lab_diag, boost.gamma0,
                                                  fieldtypes=['rho','E','J'],
                                                  period=write_period,
                                                  fldobject=sim.fld,
                                                  comm=sim.comm,
                                                  write_dir=write_dir+'/diags'),
                  BackTransformedParticleDiagnostic( zmin, zmax, v_window,dt_lab_diag_period,
                                                    N_lab_diag, boost.gamma0,
                                                    period=write_period,
                                                    fldobject=sim.fld,
                                                    select={'uz':[2,None]},
                                                    species={"electrons":plasma_elec_Inj,
                                                             "electrons_from_N":elec_from_N},
                                                    comm=sim.comm,
                                                    write_dir=write_dir+'/diags')]
    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)
    ### Run the simulation
    sim.step( N_step )
    print('Ending FBPIC simulation',flush=True)

# %%
