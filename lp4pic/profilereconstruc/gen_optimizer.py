import numpy as np
from .utils import __Error__,__AberrationRandomExplorer__
from ..focusing.Focusing import BeamFocusing
from ..utils.utils import ReadImages, Mirror,\
                    WorkingPlaneTransform, __Interpolator__, color_txt
import os, warnings
import h5py

def __Mutation__(j,N_trials,delta,A0,V0,
                 beam,image_exp,X,Y,
                 loss_func):
    print(f'--Trial {j+1:d}/{N_trials:d}, with{ delta=:e}',flush=True,end='\r')
    V = np.random.rand(len(A0))-.5
    V /= np.sqrt(V.dot(V))
    A0_mutated = np.zeros_like(A0)
    A0_mutated[1:] = (A0 + delta*(.5*V0+V))[1:]
    beam.add_aberrations(A0_mutated)
    tmp_im = beam.propagation()
    int_reb = __Interpolator__((beam.y,beam.x),tmp_im,X,Y)
    CurrentErr = __Error__(np.sqrt(int_reb),np.sqrt(image_exp),loss_func)
    return A0_mutated, CurrentErr

def GeneticOptimizer(ImageNF:ReadImages,ImageFF:ReadImages,
                     mirror:Mirror, lambda0:float, NzR:int,
                     N_evos:int, N_trials:int, best_err:float,
                     scan_amp_max:float, scan_amp_min:float, decay_it:int,
                     ab_rnd_search=False, ab_array=np.array([0.0]),
                     oam=0, adjust_F=True,
                     save_rnd=False,save_evo=False,
                     **kwargs):

    # Working plane definition and quality check
    beam = BeamFocusing(*WorkingPlaneTransform(ImageNF,
                                               ImageFF,
                                               mirror,
                                               lambda0,
                                               NzR,
                                               adjust_F),
                        oam=oam)
    # Choosing between RandomExploration or given AberrationArray and
    # calculating CurrentError before GeneticEvolution
    if (save_evo or save_rnd) and 'save_dir' in kwargs:
        save_dir = kwargs['save_dir']
        del kwargs['save_dir']
    elif (save_evo or save_rnd) and 'save_dir' not in kwargs:
        warnings.warn("No 'save_dir' specified. Using current directory.",
                     UserWarning)
        save_dir = '.'
    else:
        save_dir = None

    if 'limits' in kwargs:
        limits = kwargs['limits']
        del kwargs['limits']
    else:
        warnings.warn("No 'limits' specified. Using ImageFF limits as default.",
                     UserWarning)
        limits = ImageFF.limits

    if 'loss_func' in kwargs:
        loss_func = kwargs['loss_func']
        del kwargs['loss_func']
        print(f"Using {loss_func} for error evaluation.")
    else:
        warnings.warn("No 'loss_func' specified. Using 'MSE' as default.",
                     UserWarning)
        loss_func = 'MSE'


    if ab_rnd_search:
        print(color_txt('Searching for aberration coefficients...',255,255,0),
                        flush=True)
        defaults_dict = {'ab_limit':0.25,'n_max_Zer':5,'rnd_it':50}
        for elem, value in defaults_dict.items():
            if elem not in kwargs:
                warnings.warn(f"No '{elem}' specified. Using default {value = }.",
                              UserWarning)
            else:
                defaults_dict[elem] = kwargs[elem]
                del kwargs[elem]
        ab_limit = defaults_dict['ab_limit']
        n_max_Zer = defaults_dict['n_max_Zer']
        rnd_it = defaults_dict['rnd_it']
        A0, best_err, X, Y = \
            __AberrationRandomExplorer__(beam,ImageFF,
                                         n_max_Zer,
                                         ab_limit,
                                         rnd_it,
                                         limits,
                                         loss_func,
                                         save_rnd,
                                         save_dir)
    else:
        # Checking if ab_array is array-like
        if isinstance(ab_array,(list,tuple,np.ndarray)):
            A0 = np.array(ab_array)
        else:
            raise TypeError("'ab_array' must be 1darray-like")
        dy = beam.__dy__
        dx = beam.__dx__
        x = np.arange(limits['x'][0],limits['x'][1],dx)
        y = np.arange(limits['y'][0],limits['y'][1],dy) 
        X,Y = np.meshgrid(x,y)
    beam.add_aberrations(A0)

    # Initializing useful variables and saving file
    BestError_evos = list()
    It_evos = list()
    delta_arr = scan_amp_min + scan_amp_max*np.exp(-np.linspace(0,decay_it,decay_it))
    Ndelta = 0
    delta = delta_arr[Ndelta]
    V0 = 0*A0
    AcceptedIts = 0
    if save_evo:
        os.makedirs(save_dir,exist_ok=True)
        with h5py.File(save_dir+'/evo_data_opt.h5','w') as data_collector:
            data_collector.create_group('images')
            data_collector.create_group('Aberration_arrays')
    # Genetic evolution loop
    image_exp = ImageFF.evaluate(X,Y)

    for i in range(N_evos):
        print(f'Evolution {i+1:d}/{N_evos:d}',flush=True)
        if i%10 == 0:
            Ndelta = 0
            print('**Increasing the scan radius after 10 evolutions**\n',
                  f' Now N={Ndelta:d} and {delta=:e}.\n',
                  f' Next rejection will update delta={delta_arr[1]:e}',
                  f' with N={Ndelta+1:d}',flush=True)
        # Mutation loop
        A_mutations = list()
        Err_mutations = list()
        for j in range(N_trials):
            A0_mutated, CurrentErr = __Mutation__(j,N_trials,delta,A0,V0,
                                             beam,image_exp,X,Y,loss_func)
            A_mutations.append(A0_mutated)
            Err_mutations.append(CurrentErr)
        min_ind = np.argmin(Err_mutations)
        A = A_mutations[min_ind]
        err = Err_mutations[min_ind]
        # Rejection check on the best mutation
        # --Accepting best mutation as evolution
        if err < best_err:
            print(color_txt('Evolution accepted.',0,255,0),flush=True)
            best_err = err
            # New variation direction
            V0 = A-A0
            V0 /= np.sqrt(V0.dot(V0))
            A0 = A
            BestError_evos.append(best_err)
            It_evos.append(i)
            AcceptedIts += 1
            if save_evo:
                beam.add_aberrations(A0)
                im = beam.propagation()
                with h5py.File(save_dir+'/evo_data_opt.h5','a') as data_collector:
                    g1 = data_collector.get('images')
                    g1.create_dataset(str(i).zfill(len(str(N_evos))+1),data=im)
                    g2 = data_collector.get('Aberration_arrays')
                    g2.create_dataset(str(i).zfill(len(str(N_evos))+1),data=A0)
        # --Rejecting evolution
        else:
            print('\n\033[91m','Evolution rejected.','\033[00m',flush=True)
            Ndelta += 1
            # Renew delta
            if Ndelta < decay_it:
                print("\033[30;41m",f"Updating mutations radius with N={Ndelta:d}","\033[00m",flush=True)
                delta = delta_arr[Ndelta]
            else:
                Ndelta = 0
                print("\033[30;41m",f"Updating mutations radius with N={Ndelta:d}","\033[00m",flush=True)
                delta = delta_arr[Ndelta]
    if save_evo:
        with h5py.File(save_dir+'/evo_data_opt.h5','a') as data_collector:
            g1 = data_collector.get('images')
            g1.attrs.create('extent',data=[beam.x_lims,beam.y_lims])
            g2 = data_collector.create_group('algorithm_data')
            g2.create_dataset('best_errors',data=BestError_evos)
            g2.create_dataset('iterations_accepted',data=It_evos)
            g2.attrs.create('number_of_accepted_its',data=f'{AcceptedIts}/{N_evos}')
    beam.add_aberrations(A0)
    return beam, It_evos, BestError_evos
    