# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:10:37 2020

@author: Steffen Jung
"""

import torch
import numpy as np
import os

##############################################################################
class SpectralTransform(torch.nn.Module):
    """
    Class implementing azimuthal spectral transformation.
    The transformation is runtime efficient, but not memory efficient.
    For more information, see:
    "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions",
    https://arxiv.org/abs/2007.08457

    Methods
    -------
    fft(data:torch.tensor) -> torch.tensor
        Expects color or grayscale images. An input tensor of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        Returns the fft transformed tensor of shape (batch_size, rows, ceil(cols/2)).
        Useful for plotting.
        
    spectral_vector(data:torch.tensor) -> torch.tensor
        Expects color or grayscale images. An input tensor of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        Returns the azimuthed spectral profile of the given tensor; a vector for each image given.
        Resulting shape: (batch_size, self.vector_length)
        Useful for deep fake detection.
        
    Attributes
    -------
    r_max : int
        Index of largest radius.
        
    vector_length : int
        Length of the resulting spectral vectors.
        
    mask : torch.tensor
        Tensor of shape (self.r_max, rows, ceil(cols/2)).
        A mask for each radius. Contains 1 if the corresponding coordinates are on the radius, 0 otherwise.
        
    mask_n : torch.tensor
        Tensor of shape (self.r_max).
        Contains the number of coordinates for each radius.
    """
    
    ##########################################################################
    def __init__(self, rows:int=64, cols:int=64, eps:float=1E-8, device:str=None, mask_dtype:torch.dtype=torch.float16):
        """
        Create a module implementing spectral azimuthal transformation.
        For more information, see:
        "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions",
        https://arxiv.org/abs/2007.08457

        Parameters
        ----------
        rows : int = 64
            Size of the input tensor.
            
        cols : int = 64
            Size of the input tensor.
            
        eps : float = 1E-8
            Added during fft before taking log.
            
        device : str = None
            Which device to use.
            
        mask_dtype : torch.dtype = torch.float16
            Specify the dtype of the precomputed masks. Try to use half precision to speed up the transformation.
        """
        super(SpectralTransform, self).__init__()
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((rows,cols_onesided)) - np.array([[[shift_rows]],[[0]]])
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r,axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(
            r_max+1,-1,-1
        )
        radius_to_slice = torch.arange(r_max+1).view(-1,1,1)
        # generate mask for each radius
        mask = torch.where(
            r==radius_to_slice,
            torch.tensor(1, dtype=torch.float),
            torch.tensor(0, dtype=torch.float)
        )
        # how man entries for each radius?
        mask_n = torch.sum(mask, axis=(1,2))
        mask = mask.unsqueeze(0) # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1/mask_n).unsqueeze(0)
        # use half precision for masks
        mask = mask.to(mask_dtype)
        self.r_max = r_max
        self.vector_length = r_max+1
        
        self.register_buffer("mask", mask)
        self.register_buffer("mask_n", mask_n)
            
        if device is not None:
            self.to(device)
        self.device = device
        
        self.forward = self.spectral_vector
        
    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################
    
    ##########################################################################
    def fft(self, data:torch.tensor) -> torch.tensor:
        """
        Expects color or grayscale images. An input tensor of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        Returns the fft transformed tensor of shape (batch_size, rows, ceil(cols/2)).
        Useful for plotting.

        Parameters
        ----------
        data : torch.tensor
            The input tensor.
            Expected to be of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        
        Returns
        ----------
        torch.tensor : fft transformed input data.
            Resulting shape is (batch_size, rows, ceil(cols/2)).
        """
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data =  0.299 * data[:,0,:,:] + \
                    0.587 * data[:,1,:,:] + \
                    0.114 * data[:,2,:,:]
        # returns complex tensor
        fft = torch.rfft(data,2,onesided=True)
        # abs of complex tensor
        fft_abs = torch.sum(fft**2,dim=3)
        fft_abs = fft_abs + self.eps
        # scale for improved visibility
        fft_abs = 20*torch.log(fft_abs)
        
        return fft_abs
    
    ##########################################################################
    def spectral_vector(self, data:torch.tensor) -> torch.tensor:
        """
        Expects color or grayscale images. An input tensor of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        Returns the azimuthed spectral profile of the given tensor; a vector for each image given.
        Resulting shape: (batch_size, self.vector_length)
        Useful for deep fake detection.

        Parameters
        ----------
        data : torch.tensor
            The input tensor.
            Expected to be of shape (batch_size, channels, rows, cols) or (batch_size, rows, cols).
        
        Returns
        ----------
        torch.tensor : spectrally transformed input data.
            Resulting shape is (batch_size, self.vector_length).
        """
        fft = self.fft(data) \
                .unsqueeze(1) \
                .expand(-1,self.vector_length,-1,-1) # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2,3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1,1)
        profile = profile / profile.max(1)[0].view(-1,1)
        
        return profile
   
    

###############################################################################

        

if __name__ == "__main__":
    ffhq_real_path = "/BS/spectral-gan/nobackup/data/ffhq/64"
    file_real      = "cache_ffhq.profiles.npy"
    device         = "cuda:0"
    batch_size     = 128
    
    spectral_transform = SpectralTransform(rows=64, cols=64, device=device)
    
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from tqdm import tqdm
    
    ############################################################
    #                                                          #
    #                      transform_directory                 #
    #                                                          #
    ############################################################
    
    def transform_directory(path, file_cache, device, batch_size):
        if not os.path.isfile(file_cache):
            dataset = dset.ImageFolder(
                root = path,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            )
            
            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = False,
            )
            
            profiles = []
            
            with tqdm(total=len(dataset), desc="Fitting profiles", unit="img") as pbar:            
                for i, data in enumerate(dataloader):
                    d = data[0].to(device)
                    
                    profiles.append(
                        spectral_transform(d).detach().cpu().numpy()
                    )
                    
                    pbar.update(len(d))
                    del d
                    del data
                    
            profiles = np.concatenate(profiles)
            np.save(file_cache, profiles)
        
        else:
            profiles = np.load(file_cache)
            
        return profiles
            
    ############################################################
    #                                                          #
    #                      Compute Profiles                    #
    #                                                          #
    ############################################################
    
    reals = transform_directory(ffhq_real_path, file_real, device, batch_size)
    
    ############################################################
    #                                                          #
    #                            Plot                          #
    #                                                          #
    ############################################################
    
    import matplotlib.pyplot as plt
    plt.plot(
        np.mean(reals,axis=0).flatten()
    )
    plt.xlabel("Frequency")
    plt.ylabel("Power spectrum")
    plt.title("FFHQ Real Average Spectral Profile")
    plt.savefig("avg_profile.png")