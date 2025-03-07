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
                     "D" : 1.4})
    brush_2 = b.Brush("simulation",
                    {"filename": "data/Density_Profile_D140.csv"})
    plt.plot(brush.z, brush.phi,'k')
    plt.plot(brush_2.z, brush_2.phi,'k:')
    plt.xlim([0,100])
    plt.show()
    # Load a brush profile

    # Compute the osmotic pressure in the brush profile

    # Compute particle insertion force/energy

    # 
    

if __name__ == "__main__":
    main()