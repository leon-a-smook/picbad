# Common imports
import numpy as np
import pandas as pd
from scipy.special import gamma

# Custom imports
import src.utils as utils
import src.config as config

# Testing
import matplotlib.pyplot as plt

class Brush():
    """A class that contains all relevant parameters of a polymer brush."""
    def __init__(self,
                 profile_type: str = "model", # or "simulation"
                 params: dict = {"type" : "gaussian",
                                 "Mn" : 100,
                                 "sigma" : 1},
                 grafting_density: float = 0.1, 
                 nu: float = 0.588, 
                 kT: float = 1.0,
                 monomer_size: float = 1.0,
                 osmotic_prefactor: float = 1.0):
        
        # Default parameters
        self.N_MAX = 1e3
        self.N_RANGE = np.arange(0,config.N_MAX+1,step=1)
        self.DENSITY_PROFILE_RESOLUTION = 0.1

        # Set general brush parameters
        self.grafting_density = grafting_density
        self.nu = nu
        self.kT = kT
        self.monomer_size = monomer_size
        self.osmotic_prefactor = osmotic_prefactor
        self.has_profile = False

        # Load or generate density profile
        if profile_type not in ["model","simulation"]:
            raise(ValueError("Invalid profile type. Use model or simulation."))
        elif profile_type == "model":
            self.generate_profile(params)
        elif profile_type == "simulation":
            self.load_profile(params)

        # Generate osmotic pressure profile
        self.phi_N = self.phi/((4/3)*np.pi*(self.monomer_size/2)**3) 
        self.osm_press = self.osmotic_prefactor*(kT/self.monomer_size**3)*(self.phi_N)**(3*self.nu/(3*self.nu-1))

    def generate_profile(self,params):
        utils.validate_distribution_dict(params)
        # Select the right chain length distribution
        if params["type"] =="gaussian":
            Mn, sigma = params['Mn'], params['sigma']
            pdf = lambda N: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(N - Mn)**2/sigma**2)
        elif params["type"] == "schulz-zimm":
            Mn, D = params['Mn'], params['D']
            shape = 1/(D - 1)
            scale = Mn/shape
            pdf = lambda N: N**(shape-1)*np.exp(-N/scale)/(scale**shape * gamma(shape))
        
        # Compute the survival function
        sf = 1 - np.cumsum(pdf(self.N_RANGE))/np.sum(pdf(self.N_RANGE))
        
        # Compute corresponding volume densities
        d0 = 1/np.sqrt(self.grafting_density)
        phi = ((self.monomer_size/d0)*sf**(1/2))**(3-1/self.nu)

        # Compute corresponding height values
        z = np.zeros_like(self.N_RANGE)
        for i, _ in enumerate(self.N_RANGE):
            z[i] = d0**(1-1/self.nu)*self.monomer_size**(1/self.nu)*np.trapz(sf[:i]**(1/(2*self.nu)-1/2),self.N_RANGE[:i])

        # Resample to get equidistant points
        z_resampled = np.arange(0,np.max(z)+self.DENSITY_PROFILE_RESOLUTION,step=self.DENSITY_PROFILE_RESOLUTION)
        phi_resampled = np.interp(z_resampled,z,phi)

        self.z = z_resampled
        self.phi = phi_resampled

    def load_profile(self,params):
        utils.validate_load_profile_dict(params)
        particle_size = (4/3)*np.pi*(self.monomer_size/2)**3
        data = pd.read_csv(params["filename"])
        z = data.z.to_numpy()
        phi = data.F_mean.to_numpy()*particle_size

        # Resample data if data is not uniform
        if np.std(np.diff(z)) > 0:
            z_resampled = np.arange(0,np.max(z)+self.DENSITY_PROFILE_RESOLUTION,step=self.DENSITY_PROFILE_RESOLUTION)
            phi_resampled = np.interp(z_resampled,z,phi)
            self.z = z_resampled
            self.phi = phi_resampled
        else:
            self.z = z
            self.phi = phi