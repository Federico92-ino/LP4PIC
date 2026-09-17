import cv2
import numpy as np
import sys
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.constants import pi
from types import FunctionType
from functools import wraps
import warnings

def __Interpolator__ (yx:tuple,img,X,Y):
    interp = RegularGridInterpolator(yx,
                                     img,
                                     bounds_error=False,
                                     fill_value=0.)
    return interp((Y,X))

def __propagator__(f,dx,dy,z,kz):
    ny,nx = f.shape
    kx = 2*pi*fftshift(fftfreq(nx,dx))
    ky = 2*pi*fftshift(fftfreq(ny,dy))
    Kx,Ky = np.meshgrid(kx,ky)
    f_hat0 = fftshift(fft2(f,norm='ortho'))
    f_hatz = f_hat0*np.exp(-1j*(Kx**2+Ky**2)/kz*.5*z)
    f_z = ifft2(ifftshift(f_hatz),norm='ortho')
    return f_z

def __wrap_fit_func__(func):
   @wraps(func)
   def wrapper(*args,**kwargs):
      return func(*args,**kwargs).flatten()
   return wrapper

def color_txt(txt:str,r=128,g=128,b=128):
    string = f"\x1b[38;2;{r};{g};{b}m{txt}\x1b[00m"
    return string

def find_COM_and_Bdiam(img,blur_img,threshold,calibration,
                       com_mode,remove_bg,SGfilt_order):

    _,thresh = cv2.threshold(blur_img,
                             threshold,
                             1,
                             cv2.THRESH_BINARY)
    thresh = cv2.normalize(thresh,None,0,1,cv2.NORM_MINMAX,0)
    cnts, hierarchies = cv2.findContours(thresh,
                               cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
    cX,cY = [list(),list()]
    r_mean = list()
    R_mean = list()
    for cnt in cnts:
	# compute the center of the contour
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cX.append(cx)
            cY.append(cy)
            diff = cnt[:,0]-np.array([cx,cy])
            Radius = np.sqrt(diff[:,0]**2+diff[:,1]**2)
            radius = np.sqrt((diff[:,0]*calibration['dx'])**2
                             +
                             (diff[:,1]*calibration['dy'])**2)
            r_mean.append(radius.mean())
            R_mean.append(Radius.mean())
        else:
            r_mean.append(np.nan)
            R_mean.append(np.nan)
            cX.append(np.nan)
            cY.append(np.nan)
    R_mean = np.array(R_mean)
    r_mean = np.array(r_mean)
    parents = np.where(hierarchies[0,:,2] != -1)[0]
    if parents.size>0:
        R_parent = np.nanmax(R_mean[parents])
        D = 2*np.nanmax(r_mean[parents])
        parent_id = np.where(R_mean == R_parent)[0]
        sons = np.where(hierarchies[0,:,3] == parent_id)[0]
        R_son = np.nanmax(R_mean[sons])
        eldest_id = np.where(R_mean == R_son)[0]
        try:
            grand_sons = np.where(hierarchies[0,:,3] == eldest_id)[0]
            while grand_sons.size>0:
                R_grandson = np.nanmax(R_mean[grand_sons])
                youngest_id = np.where(R_mean == R_grandson)[0]
                grand_sons = np.where(hierarchies[0,:,3] == youngest_id)[0]
        except:
            pass
        match com_mode:
            case "imageCOM":
                M = cv2.moments(blur_img)
                n0x = int(M["m10"] / M["m00"])
                n0y = int(M["m01"] / M["m00"])
            case "max":
                n0y,n0x = np.unravel_index(np.argmax(blur_img),blur_img.shape)
            case "parent":
                n0x = cX[parent_id[0]]
                n0y = cY[parent_id[0]]
            case "son":
                n0x = cX[eldest_id[0]]
                n0y = cY[eldest_id[0]]
            case "grandson":
                if 'youngest_id' in locals():
                    n0x = cX[youngest_id[0]]
                    n0y = cY[youngest_id[0]]
                else:
                    warnings.warn("\033[38;2;255;128;0m"
                                  f"Warning: there are no 'grandsons'! "
                                  f"Using 'imageCOM' instead as COM mode."
                                  "\033[0m",Warning)
                    M = cv2.moments(blur_img)
                    n0x = int(M["m10"] / M["m00"])
                    n0y = int(M["m01"] / M["m00"])
            case _:
                warnings.warn("\033[38;2;255;128;0m"
                              f"Warning: {com_mode} is not a valid COM mode. "
                              f"Using 'imageCOM' instead.\033[0m",Warning)
                M = cv2.moments(blur_img)
                n0x = int(M["m10"] / M["m00"])
                n0y = int(M["m01"] / M["m00"])
    else:
        R_parent = np.nanmax(R_mean)
        D = 2*np.nanmax(r_mean)
        parent_id = np.where(R_mean == R_parent)[0]
        match com_mode:
            case "imageCOM":
                M = cv2.moments(blur_img)
                n0x = int(M["m10"] / M["m00"])
                n0y = int(M["m01"] / M["m00"])
            case "max":
                n0y,n0x = np.unravel_index(np.argmax(blur_img),blur_img.shape)
            case "parent":
                n0x = cX[parent_id[0]]
                n0y = cY[parent_id[0]]
            case "son"|"grandson":
                warnings.warn("\033[38;2;255;128;0m"
                              f"Warning: No 'parent' can be defined. "
                              f"{com_mode} is not a valid COM mode."
                              f"Using 'imageCOM' instead.\033[0m",Warning)
                M = cv2.moments(blur_img)
                n0x = int(M["m10"] / M["m00"])
                n0y = int(M["m01"] / M["m00"])
            case _:
                warnings.warn("\033[38;2;255;128;0m"
                              f"Warning: {com_mode} is not a valid COM mode. "
                              f"Using 'imageCOM' instead.\033[0m",Warning)
                M = cv2.moments(blur_img)
                n0x = int(M["m10"] / M["m00"])
                n0y = int(M["m01"] / M["m00"])
    com_xp,com_yp = (cX[parent_id[0]],cY[parent_id[0]])
    com_x, com_y = (n0x, n0y)
    if remove_bg == 'masking':
        img = cv2.bitwise_and(img,img,mask=thresh)
    elif remove_bg == 'filtering':
        y,x = [np.arange(0,N,dtype=int) for N in img.shape]
        X,Y = np.meshgrid(x,y)

        filt = np.exp(-2*(((X-com_xp)/R_parent)**2+((Y-com_yp)/R_parent)**2)**SGfilt_order)
        img = img*filt
    elif remove_bg is None:
        pass
    return com_x,com_y,D,img,parents                               

def center_n_crop(img,blur_img,threshold,calibration,
                  com_mode,remove_bg,SGfilt_order):
    ny,nx = img.shape
    n0x,n0y,diam,img,parents = find_COM_and_Bdiam(img,
                                          blur_img,
                                          threshold,
                                          calibration,
                                          com_mode,
                                          remove_bg,
                                          SGfilt_order)
    shift = np.array((int(ny/2-n0y+.5),int(nx/2-n0x+.5)))
    #center
    img = np.roll(a=img,
                 shift=shift,
                 axis=(0,1)
                 )
    #crop
    shift = np.abs(shift)
    img = img[shift[0]:ny-shift[0],shift[1]:nx-shift[1]]
    return n0x, n0y, diam, img,parents

class ReadImages(object):
    def __init__(self,img_path,calibration,normalize=True,threshold=0.0,
                 com_mode='imageCOM',remove_bg:None|str=None,mask_bg:None|float=None,
                 SGfilt_order=1,blur_dict={'blur_it':1,'ksize':(3,3),'sigma':0},
                 change_res=1.,square_img=True,offset={'x':0.,'y':0.},
                 center_image=True,activate_blur=False):
        self.__mask_bg__ = mask_bg
        im = cv2.imread(img_path,(cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH))

        if normalize:
            im = cv2.normalize(im,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        else:
            if im.dtype.kind in 'ui':
                info = np.iinfo(im.dtype)
                scale_factor = info.max
            else:
                scale_factor = 1
            im = im/scale_factor
        blur_it = blur_dict['blur_it']
        ksize = blur_dict['ksize']
        sigma = blur_dict['sigma']
        blur_im = im.copy()
        j = 0
        while j < blur_it:
            blur_im = cv2.GaussianBlur(blur_im,ksize,sigma)
            j +=1       
        if activate_blur:
            im = blur_im.copy()
        if center_image:
            com_x,com_y,diam,img,self.parents = center_n_crop(im,
                                                 blur_im,
                                                 threshold,
                                                 calibration,
                                                 com_mode,
                                                 remove_bg,
                                                 SGfilt_order)
        else:
            com_x,com_y,diam,img,self.parents = find_COM_and_Bdiam(im,
                                                      blur_im,
                                                      threshold,
                                                      calibration,
                                                      com_mode,
                                                      remove_bg,
                                                      SGfilt_order)
        self.CenterOfMassIndx = {'nx':com_x,'ny':com_y}
        cny,cnx = img.shape
        dx,dy = calibration['dx'],calibration['dy']
        #x-array of centered image
        cx = np.arange(0,cnx)*dx
        cx = cx - np.mean(cx)
        #y-array of centered image
        cy = np.arange(0,cny)*dy
        cy = cy - np.mean(cy)
        cy = np.flip(cy)
        px_aspect_ratio = dy/dx
        if px_aspect_ratio >= 1:
            nx = cnx
            ny = int(cny*px_aspect_ratio)
            new_x = np.linspace(cx.min(),cx.max(),int(nx*change_res))
            new_y = np.linspace(cy.min(),cy.max(),int(ny*change_res))
        else:
            nx = int(cnx/px_aspect_ratio)
            ny = cny
            new_x = np.linspace(cx.min(),cx.max(),int(nx*change_res))
            new_y = np.linspace(cy.min(),cy.max(),int(ny*change_res))
        if square_img:
            ctrl = new_x.max()-new_y.max()
            if ctrl>=0:
                new_y = new_x
            else:
                new_x = new_y
        if center_image:
            self.CenterOfMassIndx['nx'] = np.where(new_x<=0.)[0][-1]+1
            self.CenterOfMassIndx['ny'] = np.where(new_y<=0.)[0][-1]+1
        else:             
            self.CenterOfMassIndx['nx'] = np.where(new_x<=cx[com_x])[0][-1]
            self.CenterOfMassIndx['ny'] = np.where(new_y<=cy[com_y])[0][-1]
        NewX, NewY = np.meshgrid(new_x,new_y)
        new_img = __Interpolator__((cy,cx),
                                   img,
                                   NewX+offset['x'],
                                   NewY+offset['y'])

        self.x,self.y = new_x, new_y
        self.__dx__,self.__dy__ = [a[1]-a[0] for a in [self.x,self.y]]
        if self.__mask_bg__ is not None:
            self.image = np.ma.masked_less_equal(new_img,self.__mask_bg__*sys.float_info.epsilon)
        else:
            self.image = new_img
        self.image_charge = np.trapz(np.trapz(self.image,
                                              dx=self.__dx__),
                                     dx=self.__dy__)
        self.limits = {'x':[self.x.min(),self.x.max()],
                       'y':[self.y.min(),self.y.max()]}
        self.estimated_beam_diameter = diam

    def evaluate(self,x,y):
        values = __Interpolator__((self.y,self.x),self.image,x,y)
        if self.__mask_bg__:
            return np.ma.masked_less_equal(values,self.__mask_bg__*sys.float_info.epsilon)
        else:
            return values

    def renormalize(self,norm):
        self.image = norm*self.image/self.image_charge
        self.image_charge = np.trapz(np.trapz(self.image,
                                              dx=self.__dx__),
                                     dx=self.__dy__)

    def edges_cut(self,xmin,xmax,ymin,ymax):
        X,Y = np.meshgrid(self.x,self.y)
        y_inds, x_inds = np.where((Y>=ymin)&(Y<=ymax)&(X>=xmin)&(X<=xmax))
        self.image = self.image[y_inds.min():y_inds.max(),
                                x_inds.min():x_inds.max()]
        self.image_charge = np.trapz(np.trapz(self.image,
                                              dx=self.__dx__),
                                     dx=self.__dy__)
        self.x = self.x[x_inds.min():x_inds.max()]
        self.y = self.y[y_inds.min():y_inds.max()]
        self.limits = {'x':[self.x.min(),self.x.max()],
                       'y':[self.y.min(),self.y.max()]}

    def SuperGaussFit(self,guess:list|tuple|np.ndarray,mode:str='bi',
                      **kwargs):
        if 'p0' in kwargs:
            del kwargs['p0']
        data2fit = self.image.flatten()
        xy = (self.x,self.y)
        if mode == 'mono':
            @__wrap_fit_func__
            def fit_func(xy,A0,x0,y0,sigma_x,order=1):
                """
                Docstring per fit_func
                
                :param xy: tuple of x,y arrays
                :param A0: Amplitude
                :param x0: centroid in x dir
                :param y0: centroid in y dir
                :param sigma_x: waist
                :param order: SuperGaussian order
                """
                x,y = np.meshgrid(xy[0],xy[1])
                rho = np.sqrt(x**2+y**2)
                rho0 = np.sqrt(x0**2+y0**2)
                rho_prime = (rho-rho0)
                data = A0*np.exp(-2*(((rho_prime/sigma_x)**2)**order))
                return data
        elif mode == 'bi':
           @__wrap_fit_func__
           def fit_func(xy,A0,x0,y0,sigma_x,sigma_y,theta=0,order=1):
              """
              Docstring per fit_func
              
              :param xy: tuple of x,y arrays
              :param A0: Amplitude
              :param x0: centroid in x dir
              :param y0: centroid in y dir
              :param sigma_x: axis of the ellipse in x dir
              :param sigma_y: axis of the ellipse in y dir
              :param theta: tilt with respect to the cartesian coordinates
              :param order: SuperGaussian order
              """
              x,y = np.meshgrid(xy[0],xy[1])
              x_prime = (x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)
              y_prime = (y-y0)*np.cos(theta) - (x-x0)*np.sin(theta)
              data = A0*np.exp(-2*(((x_prime/sigma_x)**2+(y_prime/sigma_y)**2)**order))
              return data
        else:
            raise ValueError(f"'{mode}' is not a fitting mode!")
        output = curve_fit(fit_func,xy,data2fit,guess,**kwargs)
        popt = output[0]
        labels = ['Amplitude','centroid','spot_size']
        match mode:
            case 'mono':
                props = [popt[0],
                         dict(zip(('x','y'),popt[1:3].tolist())),
                         popt[3]]
                if len(guess)==4:
                    props.append(1)
                else:
                    props.append(popt[4])
                labels.append('order')
            case 'bi':
                props =[popt[0],
                        dict(zip(('x','y'),popt[1:3].tolist())),
                        dict(zip(('x','y'),popt[3:5].tolist()))]
                if len(guess)==5:
                   props.extend((0,1))
                elif len(guess)==6:
                   props.extend((popt[5]%pi,1))
                elif len(guess)>6:
                   props.extend((popt[5]%pi,popt[6]))
                labels.extend(('angle','order'))
        self.fit_params_results = {'mode':mode,'opt_params':dict(zip(labels,props))}
        self.fit_func = fit_func.__wrapped__
        return output

class Mirror(object):
    def __init__(self,F_dist,shape:str|FunctionType):
        self.focal_distance = F_dist
        self.shape = shape
    def __MirrorPhase__(self,R,kz):
        shape = self.shape
        F = self.focal_distance
        if isinstance(shape,str):
            match shape:
                case 'Parabolic':
                    phase = -kz*R**2/(2*F)
                case 'Spherical':
                    phase = 2*kz*np.sqrt(4*F**2-R**2)
        elif callable(shape):
            phase = shape(R,kz,F)
        else:
            raise ValueError("MirrorShape must be either "\
                "Parabolic or Spherical or a custom function(R,kz,F)")
        return phase

class ResolutionWarning(Warning):
   def __init__(self,message):
      self.message = message
   def __str__(self):
      return str(self.message)

def WorkingPlaneTransform(NearField:ReadImages,
                          FarField:ReadImages|None,
                          mirror:Mirror,
                          lambda0,
                          NzR,
                          adjust_F:bool=False):
    B_diam = NearField.estimated_beam_diameter
    FN = mirror.focal_distance/B_diam
    waist = FN*lambda0
    zR = np.pi*(waist)**2/lambda0
    F = NzR*zR
    s = F/mirror.focal_distance
    #Resolution check of the nominal waist
    dxw = NearField.__dx__*s
    Nw = int(waist/dxw+.5)
    if FarField is not None:
        wFF = FarField.estimated_beam_diameter/2
        dxFF = FarField.__dx__
        NwFF = int(wFF/dxFF+.5)
        if Nw < NwFF:
            dxw = waist/NwFF
            warnings.warn(color_txt(f"Ehi watch out!\n"+\
                                    "At such distance the waist resolution "+\
                                    f"is approximately {Nw} pixels,"+\
                                    f"\nless than experimental farfield resolution {NwFF}.",
                                    250,20,20),
                         ResolutionWarning)
            sys.stderr.flush()
            if adjust_F:
                print(f"\nThe effective focal distance F = {F/zR:.1f}*RayleighRange may be too long",
                f"\n and will be adjusted to F = {mirror.focal_distance*dxw/(NearField.__dx__*zR):.1f}*RayleighRange.",
                f"\nIf you want to force this beaviour, please set 'adjust_F'=False.",
                flush=True)

                s = dxw/NearField.__dx__
                F = mirror.focal_distance*s
        else:
            print(color_txt(f"With effective focal length "+\
                            f"F = {F/zR:.1f}*RayleighRange\n"+\
                            f"the resolution of the nominal waist is {Nw} pixels.",
                            20,250,20),
                 flush=True)
    else:
        print(color_txt(f"With an estimated focal spot size of {waist*1e6:.2f} um "+\
                        f"and a Rayleigh range of {zR*1e3:.2f} mm,\n"+\
                        f"the effective focal distance is F = {F*1e2:.1f} cm.",250,250,20),
             flush=True)

    WP_mirror = Mirror(F,mirror.shape)
    WP_xlims = np.array(NearField.limits['x'])*s
    WP_ylims = np.array(NearField.limits['y'])*s
    WP_norm_radius = B_diam/2*s
    image = NearField.image/s**2
    return WP_mirror,lambda0,image,\
           WP_xlims,WP_ylims,WP_norm_radius