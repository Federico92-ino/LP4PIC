import numpy as np
from skimage.restoration import unwrap_phase
from skimage.filters import gaussian
import os,copy
import time
import h5py
from typing import Sequence
from ..focusing import GenerateZerBank,ZerNorm,BeamFocusing
from ..utils.utils import  ReadImages, Mirror,WorkingPlaneTransform,\
                     color_txt, __propagator__
from .utils import  __Error__, __max_align_check__, L2_product, __Interpolator__


class PhaseReconstructor(object):
    def __init__(self,NearField:ReadImages,FarField:Sequence[ReadImages],
                 mirror:Mirror,lambda0:float, planes_pos:Sequence[float]=[0.0],
                 phase_mask:None|float|np.ndarray=None,seed_phase:float|np.ndarray=0,
                 save_dir=None):
        self.NearField = NearField
        self.FarField = FarField
        self.planes_pos = planes_pos
        self.Mirror = mirror
        self.lambda0 = lambda0
        self.phase_mask = phase_mask
        if isinstance(seed_phase,(int,float)):
            self.__seed__ = seed_phase
            self.seed_phase = seed_phase*np.ones_like(NearField.image)
        elif isinstance(seed_phase,np.ndarray):
            if seed_phase.shape != NearField.image.shape:
                raise ValueError("The seed phase has to be"\
                                 " as shaped as NearField")
            else:
                self.seed_phase = seed_phase
        self.__wrapped_phase__ = self.seed_phase
        #self.retrieved_phase = self.seed_phase
        self.save_dir = save_dir
        self.__timebar__ = ['|','/','-','\\']
        if self.save_dir:
            os.makedirs(self.save_dir,exist_ok=True)
            with h5py.File(self.save_dir+"/data_gs.h5",'a') as data_collector:
                try:
                    data_collector.create_group('intensity')
                except:
                    pass
                try:
                    data_collector.create_group('phase')
                except:
                    pass

    def __seq_roundtrip__(self,n,N,fNF,F,kz,mirror_phase,dx,dy,
                            X,Y,sigma,truncate,align_max,loss_func):
        #1) NF Candidate build up and first propagation
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}",
                        int((1-n/N)*255),255,0),
             end='\033[2K\r',flush=True)
        f_nf_ = fNF*np.exp(1j*(self.__wrapped_phase__+mirror_phase))
        dz = self.planes_pos[0]
        f_ff = __propagator__(f_nf_,dx,dy,F+dz,kz)
        #2a) Loop over FarField images
        if len(self.FarField) > 1:
            for i,FFimage in enumerate(self.FarField[:-1]):
                print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                                f" {n+1}/{N}:",
                                int((1-n/N)*255),255,0)+\
                      color_txt(f"  looping on FarField scan {i+1}/{len(self.FarField)}",
                               int((1-i/len(self.FarField[:-1]))*255),
                               int((i/len(self.FarField[:-1]))*255),
                               0),
                     end='\033[2K\r',flush=True)
                fFF = np.sqrt(FFimage.evaluate(X,Y))
                ## Alignment check
                if align_max:
                    rolled_FFimage = __max_align_check__(np.abs(f_ff)**2,
                                                         fFF**2,
                                                         X,Y)
                    fFF = np.sqrt(rolled_FFimage)
                ## Gaussian blurring and summing error
                f_ff = gaussian(f_ff.real,sigma,truncate=truncate)+\
                    1j*gaussian(f_ff.imag,sigma,truncate=truncate)
                self.error[n] = self.error[n]+__Error__(abs(f_ff),fFF,loss_func)
                ## Candidate build up
                f_ff_ = fFF*f_ff/np.abs(f_ff)
                dz = self.planes_pos[i+1]-self.planes_pos[i]
                f_ff = __propagator__(f_ff_,dx,dy,dz,kz)
            i+=1
        else:
            i=0
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}:",
                        int((1-n/N)*255),255,0)+\
              color_txt(f"  looping on FarField scan {i+1}/{len(self.FarField)}",
                       int((1-i/len(self.FarField))*255),
                       int((i/len(self.FarField))*255),
                       0),
             end='\033[2K\r',flush=True)
        #2b) Focus plane 
        FFimage = self.FarField[-1]
        fFF = np.sqrt(FFimage.evaluate(X,Y))
        ## Alignment check
        if align_max:
            rolled_FFimage = __max_align_check__(np.abs(f_ff)**2,
                                                 fFF**2,
                                                 X,Y)
            fFF = np.sqrt(rolled_FFimage)
        ## Gaussian blurring and summing error
        f_ff = gaussian(f_ff.real,sigma,truncate=truncate)+\
            1j*gaussian(f_ff.imag,sigma,truncate=truncate)
        self.error[n] = self.error[n]+__Error__(abs(f_ff),fFF,loss_func)
        ## Candidate build up
        f_ff_ = fFF*f_ff/np.abs(f_ff)
        #3) Back propagation to NearField
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}: back-propagating to NearField",
                        int((1-n/N)*255),255,0),
             end='\033[2K\r',flush=True)
        dz = self.planes_pos[-1]
        f_nf = __propagator__(f_ff_,dx,dy,-(F+dz),kz)
        ## Gaussian blurring and finalizing error
        f_nf = gaussian(f_nf.real,sigma,truncate=truncate)+\
            1j*gaussian(f_nf.imag,sigma,truncate=truncate)
        self.error[n] =  self.error[n]+__Error__(abs(f_nf),fNF,loss_func)
        #4) Phase retrieval
        if n+1 < N:
            print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                            f" {n+1}/{N}:         ... next iteration.",
                            int((1-n/N)*255),255,0),
                 end='\033[2K\r',flush=True)
        else:
            print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                            f" {n+1}/{N}: last round. Finish.",
                            int((1-n/N)*255),255,0),
                 end='\n',flush=True)      
        self.__wrapped_phase__ = np.angle(f_nf/np.abs(f_nf)*np.exp(-1j*mirror_phase))

    def __rnd_roundtrip__(self,n,N,fNF,F,kz,mirror_phase,dx,dy,
                            X,Y,sigma,truncate,align_max,loss_func):
        j = np.random.choice(range(len(self.FarField)),1,replace=False)[0]
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}",
                        int((1-n/N)*255),int(n/N*255),0),
             end='\033[2K\r',flush=True)
        #1) NF Candidate build up 
        f_nf_ = fNF*np.exp(1j*(self.__wrapped_phase__+mirror_phase))
        dz = self.planes_pos[j]
        #2) Propagation to random FarField plane
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}: propagating to {j+1}th FarField",
                        int((1-n/N)*255),int(n/N*255),0),
             end='\033[2K\r',flush=True)
        f_ff = __propagator__(f_nf_,dx,dy,F+dz,kz)
        fFF = np.sqrt(self.FarField[j].evaluate(X,Y))
        ## Alignment check
        if align_max:
            rolled_FFimage = __max_align_check__(np.abs(f_ff)**2,
                                                 fFF**2,
                                                 X,Y)
            fFF = np.sqrt(rolled_FFimage)
        ## Gaussian blurring and summing error
        f_ff = gaussian(f_ff.real,sigma,truncate=truncate)+\
            1j*gaussian(f_ff.imag,sigma,truncate=truncate)
        self.error[n] = self.error[n]+__Error__(abs(f_ff),fFF,loss_func)
        ## Candidate build up
        f_ff_ = fFF*f_ff/np.abs(f_ff)
        #3) Back propagation to NearField
        print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                        f" {n+1}/{N}: back-propagating to NearField",
                        int((1-n/N)*255),int(n/N*255),0),
             end='\033[2K\r',flush=True)
        f_nf = __propagator__(f_ff_,dx,dy,-(F+dz),kz)
        ## Gaussian blurring and finalizing error
        f_nf = gaussian(f_nf.real,sigma,truncate=truncate)+\
            1j*gaussian(f_nf.imag,sigma,truncate=truncate)
        #4) Phase retrieval
        if n+1 < N:
            print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                            f" {n+1}/{N}:       ... next iteration.",
                            int((1-n/N)*255),int(n/N*255),0),
                 end='\033[2K\r',flush=True)
        else:
            print(color_txt(f"{self.__timebar__[n%4]}Running iteration"+\
                            f" {n+1}/{N}: last round. Finish.",
                            int((1-n/N)*255),int(n/N*255),0),
                 end='\n',flush=True)
        self.__wrapped_phase__ = np.angle(f_nf/np.abs(f_nf)*np.exp(-1j*mirror_phase))
        self.error[n] =  self.error[n]+__Error__(abs(f_nf),fNF,loss_func)

    def GerSaxPhaseRetriever(self,NzR,N=10,focus_indx=0,
                             adjust_F=True,align_max=False,
                             mod='Sequential', mod_dim=3,
                             loss_func='MSE'):
        # Out of the loop:initialization
        self.error = np.zeros(N)
        self.__error__ = np.zeros(N)
        print(color_txt("Starting...",245,255,0),flush=True,end='\r')
        ## initializing Propagator to return
        focus_FF = self.FarField[focus_indx]
        self.Propagator = BeamFocusing(*WorkingPlaneTransform(self.NearField,
                                                        focus_FF,
                                                        self.Mirror,
                                                        self.lambda0,
                                                        NzR,
                                                        adjust_F)
                                )
        if isinstance(self.phase_mask,(float,np.ndarray)):
            print(color_txt("Applying phase mask to NearField",
                            255,200,0),flush=True)
            self.Propagator.apply_phase_mask(self.phase_mask)
        X,Y= self.Propagator.__X__,self.Propagator.__Y__
        dx = self.Propagator.__dx__
        dy = self.Propagator.__dy__
        kz = self.Propagator.__kz__
        ## Field distribution, mirror phase, effective focal distance 
        ## at input plane retrieving
        fNF = np.sqrt(self.Propagator.intensity_distribution)
        self.f_nf = fNF
        mirror_phase = self.Propagator.Mirror.__MirrorPhase__(self.Propagator.R,kz)
        F = self.Propagator.Mirror.focal_distance
        print(color_txt("Seeding the Gerchberg-Saxton loop",
                        255,255,0),flush=True,end='\n')
        # GS loop
        n = 0
        time.sleep(.5)
        print('',end='\x1b[2K\r')
        if mod == 'Test':
            self.__wrapped_phase__ = self.__seed__*np.ones_like(self.FarField[0].image)
        ## the loop
        while n < N:
            if mod == 'Sequential':
                ## gauss blur params
                kernel = mod_dim*np.exp(-n/N)
                sigma = 0.3*(.5*(kernel-1)-1)+.8
                truncate = int(.5*(kernel-1)/sigma-0.5)
                ## roundtrip
                self.__seq_roundtrip__(n,N,fNF,F,kz,mirror_phase,dx,dy,
                                         X,Y,sigma,truncate,align_max,loss_func)

            elif mod == 'Random':
                ## gauss blur params
                kernel = mod_dim*np.exp(-n/N)
                sigma = 0.3*(.5*(kernel-1)-1)+.8
                truncate = int(.5*(kernel-1)/sigma-0.5)
                ## roundtrip
                self.__rnd_roundtrip__(n,N,fNF,F,kz,mirror_phase,dx,dy,
                                       X,Y,sigma,truncate,align_max,loss_func)
            else:
                raise ValueError(f"{mod} is not  a recognized roundtrip modality!")
            ## saving data
            if self.save_dir:
                self.Propagator.field_distribution = fNF*np.exp(1j*self.__wrapped_phase__)
                dataI = self.Propagator.propagation()
                with h5py.File(self.save_dir+"/data_gs.h5",'a') as data_collector:
                    groupI = data_collector.get('intensity')
                    groupI.create_dataset(str(n).zfill(len(str(N))+1),data=dataI)
                    groupP = data_collector.get("phase")
                    groupP.create_dataset(str(n).zfill(len(str(N))+1),data=self.__wrapped_phase__)
            n += 1        
            print('',end='\x1b[2K\r')
        if self.save_dir:
            with h5py.File(self.save_dir+"/data_gs.h5",'a') as data_collector:
                data_collector['intensity'].attrs['error'] = self.error 
                data_collector['intensity'].attrs['extent'] = np.array([[pos for pos in limits]\
                                                                        for limits in [self.Propagator.x_lims, self.Propagator.y_lims]]).flatten()
        
        self.retrieved_phase = unwrap_phase(self.__wrapped_phase__,True)
        self.Propagator.field_distribution = fNF*np.exp(1j*self.retrieved_phase)

    def Aberrometer(self,n_rad_order:int,phase_kind:str='unwrapped'):
        MaxNoll = int((n_rad_order+1)*(n_rad_order+2)/2)
        Abarray = np.zeros(MaxNoll)
        R = self.Propagator.R/self.Propagator.__radius__
        T = self.Propagator.T
        ZerBank = GenerateZerBank(n_rad_order,R,T)
        noll = 0
        projected_phase = np.zeros_like(ZerBank[0][0])
        match phase_kind:
            case "unwrapped":
                phase = self.retrieved_phase
            case "wrapped":
                phase = self.__wrapped_phase__
        for n,Zn in ZerBank.items():
            for m,Znm in Zn.items():
                a = ZerNorm(m,n)*L2_product(phase,
                                            Znm,
                                            self.Propagator.x/self.Propagator.__radius__,
                                            self.Propagator.y/self.Propagator.__radius__)
                Abarray[noll] = a
                projected_phase += a*Znm
                noll += 1
        return Abarray, projected_phase
        

