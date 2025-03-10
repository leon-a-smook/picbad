# Common imports
import numpy as np
import pandas as pd
from scipy.special import gamma

# Custom imports
import utils
import config

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
        self.osmotic_pressure = self.osmotic_prefactor*(kT/self.monomer_size**3)*(self.phi_N)**(3*self.nu/(3*self.nu-1))

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
        sf[sf < 0] = 0 # Prevent numerical errors through negative survival functions

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

    def insert_particle(self,radius,beta=0.0):
        """This function inserts a particle into a brush with a defined radius. The effect of the 
        surface is accounted for with the parameter beta. The expressions are the same as used in
        [de Beer, Macromolecules, 2016, 49, 1070-1078]."""

        # Perpare data structures
        E_osmotic, E_surface = np.zeros_like(self.z), np.zeros_like(self.z)

        # Integrate energy contributions over particle with finite size
        for i, h in enumerate(self.z):
            r_vals = np.real(np.emath.sqrt(radius**2 - (h-self.z)**2))
            E_osmotic[i] = np.trapz(self.osmotic_pressure*np.pi*r_vals**2, self.z)
            valid_mask = self.phi_N > 0
            y = -beta*self.osmotic_pressure[valid_mask]*self.phi_N[valid_mask]**(-self.nu/(3*self.nu-1))*np.pi*r_vals[valid_mask]
            E_surface[i] = np.trapz(y,self.z[valid_mask])

        self.insertion_energy = E_osmotic + E_surface
        self.insertion_force = -np.diff(self.insertion_energy,append=0)/np.diff(self.z,append=1)
        
    def compress_profile(self,beta=0.0, redistribute_polymer=True, z_min=0.0, surface_area=1):
        """This function compresses the density profile and computes the work done and force exerted."""
        z_rev = self.z[::-1]
        # phi_rev = self.osm_press[::-1]
        phi_rev = np.copy(self.phi_N[::-1])
        # Find indices where values are NaN
        nan_indices = np.isnan(phi_rev)

        # Interpolate missing values
        phi_rev[nan_indices] = np.interp(
            np.flatnonzero(nan_indices),  # Indices of NaNs
            np.flatnonzero(~nan_indices), # Indices of non-NaNs
            phi_rev[~nan_indices]             # Corresponding non-NaN values
        )
        
        # Create a copy of density profile for redistribution
        phi_rev_og = np.copy(phi_rev)

        # Prepare data structures for results
        E_osm = np.zeros_like(phi_rev)
        E_surf = np.zeros_like(phi_rev)

        # For each height, compute the osmotic and surface energy
        dz = np.diff(self.z,append=1)
        for i, z in enumerate(z_rev):
            # Only compute for positive z-values above 0.0 and minimum 
            if z <= z_min or z <= 0.0:
                break

            # Compute osmotic work by integrating osmotic pressure slice
            osm_press = self.osmotic_prefactor*(self.kT/self.monomer_size**3)*(phi_rev[i])**(3*self.nu/(3*self.nu-1))
            delta_work = osm_press*dz[i]
            if i == 0:
                E_osm[i] = delta_work
            else:
                E_osm[i] = E_osm[i-1] + delta_work
            
            # Compute surface work (check if base > 0.0 to prevent errors)
            if phi_rev[i] > 0.0:
                E_surf[i] = -beta*osm_press*phi_rev[i]**(-self.nu/(3*self.nu-1))
            else:
                E_surf[i] = 0   
            if i == 0:
                displaced = 0
            else:
                displaced = phi_rev[i]*dz[i]
            
            phi_rev[:i] = 0

            # Redistribute polymer if requested
            if redistribute_polymer:
                phi_rev[i:] += displaced/z
        
        # Aggregate results
        self.compression_energy = E_osm[::-1] + E_surf[::-1]
        self.compression_energy = self.compression_energy*surface_area
        self.compression_force = -np.diff(self.compression_energy,append=0)/np.diff(self.z,append=1)