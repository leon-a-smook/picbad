import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.brush as b

def main():
    """
    This main function demonstrates the sequence of operations to use the theoretical
    framework in this project.

    1. Set system parameters
    2. Load or generate density profile
    3. Compute particle insertion or compression
    4. Visualize results
    """
    print("Starting analysis.")
    brush = b.Brush("model",
                    {"type": "schulz-zimm",
                     "Mn" : 100,
                     "D" : 1.4001})
    brush_2 = b.Brush("simulation",
                    {"filename": "data/Density_Profile_D140.csv"})
    
    fig, ax = plt.subplots()
    ax.plot(brush.z, brush.phi,'k')
    ax.plot(brush_2.z, brush_2.phi,'k:')
    ax.set_xlim([0,100])

    brush.compress_profile(surface_area=2250,redistribute_polymer=False)
    brush_2.compress_profile(surface_area=2250,redistribute_polymer=False)
    brush.insert_particle(1*np.sqrt(10),beta=-0.001)
    brush_2.insert_particle(1*np.sqrt(10),beta=-0.001)
    data = pd.read_csv("data/Small_Particle_D140.csv")
    x = data.z.to_numpy() - 3.0 # OFFSET BECAUSE WALL HAS THICKNESS
    y = data.F_mean.to_numpy()
    fig, ax = plt.subplots()
    ax.plot(brush.z, brush.insertion_force,label='theory')
    ax.plot(brush_2.z, brush_2.insertion_force,label='sim-based')
    ax.plot(x,y,label='simulation')
    ax.set_xlim([20,100])
    ax.set_ylim([0,3])
    ax.legend()
    plt.show()
    # Load a brush profile

    # Compute the osmotic pressure in the brush profile

    # Compute particle insertion force/energy

    # 
    

if __name__ == "__main__":
    main()