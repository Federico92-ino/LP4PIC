import numpy as np
#from scipy.interpolate import RegularGridInterpolator
import h5py
import os, warnings

from ..utils.utils import ReadImages,__Interpolator__
from ..focusing.Focusing import BeamFocusing

def __Error__(field1,field2,kind):
    match kind:
        case 'MSE':
            err = np.trapz(np.trapz((field1**2-field2**2)**2))
        case 'DKL':
            f22 = np.where(field2**2>0.,field2**2,1e-100)
            f12 = np.where(field1**2>0.,field1**2,1e-100)
            err = np.trapz(np.trapz(f22*(np.log(f22/f12))))
        case _:
            raise warnings.warn(f"There is no {kind} error metric implemented.\n"\
                                f"'MSE' will be used",UserWarning)
    return err

def __AberrationRandomExplorer__(beam:BeamFocusing,
                                 ImageFF:ReadImages,
                                 n_max_Zer,
                                 ab_limit,
                                 iterations,
                                 limits,
                                 err_kind,
                                 save_data,
                                 save_dir):
    if save_data and not save_dir:
        save_dir = './'
    #Bulding the args for Image of ff_Reb and its evaluation
    dy = beam.__dy__
    dx = beam.__dx__
    x = np.arange(limits['x'][0],limits['x'][1],dx)
    y = np.arange(limits['y'][0],limits['y'][1],dy)
    X,Y = np.meshgrid(x,y)

    ErrSeries = list()
    ASeries = list()
    if save_data:
        os.makedirs(save_dir,exist_ok=True)
        data_collector = h5py.File(save_dir+'/rnd_data_opt.h5','w')
        data_collector.close()
        del data_collector

    for i in range(iterations):
        print(f'Trial {i+1}/{iterations} of random exploration',
              flush=True,end='\r')
        MaxNoll = int((n_max_Zer+1)*(n_max_Zer+2)/2)
        A = ab_limit*(2*np.random.rand(MaxNoll)-1)
        beam.add_aberrations(A)
        tmp_intensity = beam.propagation()
        int_reb = __Interpolator__((beam.y,beam.x),tmp_intensity,X,Y)
        image_exp = ImageFF.evaluate(X,Y)
        err = __Error__(np.sqrt(int_reb),np.sqrt(image_exp),err_kind)
        ErrSeries.append(err)
        ASeries.append(A)
        if save_data:
            with h5py.File(save_dir+'/rnd_data.h5','a') as data_collector:
                data_collector.create_dataset(str(i).zfill(len(str(iterations))+1),
                                              data=tmp_intensity)

    ind = np.argmin(ErrSeries)
    err = ErrSeries[ind]
    A = ASeries[ind]
    print(f'\nMinimum departure aberration coefficients array is:\n{A = }',
          flush=True)
    if save_data:
        with h5py.File(save_dir+'/rnd_data.h5','a') as data_collector:
            data_collector.attrs.create('extent',data=[beam.x_lims,beam.y_lims])
            data_collector.attrs.create('best_iteration',data=str(ind).zfill(len(str(iterations))+1))
            data_collector.attrs.create('aberration_array',data=A)
    return A, err, X, Y

def __max_align_check__(ref_img:np.ndarray,roll_img:np.ndarray,X,Y):
    Ny,Nx = np.unravel_index(np.argmax(ref_img),ref_img.shape)
    ny,nx = np.unravel_index(np.argmax(roll_img),roll_img.shape)
    shift = [Ny-ny,Nx-nx]
    x,y = X[0,:],Y[:,0]
    roll_img = np.roll(roll_img,shift=shift,axis=(0,1))
    if shift[0] >= 0:
        match np.sign(shift[1]):
             case 0|1:
                roll_img = roll_img[shift[0]:,shift[1]:]
                x = x[shift[1]:]
                y = y[shift[0]:]
             case -1:
                roll_img = roll_img[shift[0]:,:shift[1]]
                x = x[:shift[1]]
                y = y[shift[0]:]
    else:
        match np.sign(shift[1]):
             case 0|1:
                roll_img = roll_img[:shift[0],shift[1]:]
                x = x[shift[1]:]
                y = y[:shift[0]]
             case -1:
                roll_img = roll_img[:shift[0],:shift[1]]
                x = x[:shift[1]]
                y = y[:shift[0]]
    roll_img = __Interpolator__((y,x),roll_img,X,Y)
    return roll_img

def L2_product(f,g,x,y):
    braket = np.trapz(np.trapz(f*g,x=x),x=y)
    return braket