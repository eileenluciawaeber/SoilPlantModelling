import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

import numpy as np

import json

from SALib.sample.sobol import sample

from concurrent.futures import ProcessPoolExecutor


# --------- Inputs --------- 
dx, dy, dz = 100, 100, 0.3

h_b = -1*np.arange(10,15010+100,dx, dtype=int) #bulk soil water potential in cm heads
E = 3e-6*np.arange(0,3000+1,dz) #transpiration rate in cm3 s**-1

with open('2_Data/soil_Data.json') as f:
    soil_Data = json.load(f)

soil_types_interested = ["Clay", "Silt Loam", "Sandy Loam"]
soil_Data = {soil: soil_Data[soil] for soil in soil_types_interested}

r0 = 0.05 # cm root radius in cm
V_root = np.pi*3000

nworkers = 64

results = []

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


#  --------- Model  --------- 
def monotone(curve): # to make sure that the h_leaf curve will be monotonically increasing
    curve_mono = np.minimum.accumulate(curve) # make the function monotone
    mask_mono = curve_mono == np.nanmin(curve_mono) # all values that were set to minimum

    curve_mono[mask_mono] = np.nan # set the values where curve was initially increasing to nan values

    return curve_mono

def Calc_hleaf(E, h_b, params_soil, params_plant):
    r0, h0, k0, tau = params_soil
    R_root, h0_x, tau_x, L  = params_plant

    r2 = np.sqrt(V_root/(np.pi*L))
    k0_x = 1/R_root

    c_soil = -2 * np.pi * r0 * L * k0 / (1 - tau) / (h0 ** (-tau)) / (r0 / 2 - r0 * r2 ** 2 * (np.log(r2) - np.log(r0)) / (r2 ** 2 - r0 ** 2))
    h_root = -1*np.abs(-E/c_soil + (complex(h_b))**(1-tau))**(1/(1-tau))  # Calculate root water potential from Eq.S2 and S4 using Eq.V; one value for each h_b-E pair

    #dissipation in the root
    h_x = h_root - R_root*E     # Couvreur model; from E = K_root*(h_root - h_x); R_root is the root's resistance

    # dissipation in the xylem including cavitation
    c_x = -k0_x/(1-tau_x)*(complex(h0_x)**tau_x)
    h_leaf = -1*np.abs(E/c_x+h_x.astype(np.complex128)**(1-tau_x))**(1/(1-tau_x))

    return monotone(h_leaf)

def Calc_SOL(E, hb_vect, params_soil, params_plant, gradmax = 0.7):
    SOL_hleaf = []
    SOL_E = []

    for _, hb in enumerate(hb_vect):
        h_leaf = Calc_hleaf(E, hb, params_soil, params_plant) # Calculate the iso psi soil curve
        
        FX = np.gradient(E, -h_leaf)

        rel_grad = np.abs(FX)/np.nanmax(np.abs(FX))
        pos = np.nanmax(np.where(rel_grad >= gradmax)[0]) # Find earliest position where condition holds (where gradient is larger than threshold)
        
        # add the E and h_leaf values to the SOL
        SOL_E.append(E[pos])
        SOL_hleaf.append(h_leaf[pos])
    
    return np.column_stack([hb_vect, np.array(SOL_hleaf), np.array(SOL_E)])    

def HydraulicModel(E, hb_vect, params_soil, params_plant):
    
    gradmax = 0.7 # criterion for the relative gradient - one could chose a different value (0.5, 0.8, etc... the closer it is to one, the more sensitive is the stomatal closure)

    SOL = Calc_SOL(E, hb_vect, params_soil, params_plant, gradmax=gradmax)

    return SOL


#  --------- Define Problem for SA  --------- 
variables = ['Root Resistance', 'h_0,x', 'tau xylem', 'Root Length']
num_vars = len(variables)

problem = {
    'num_vars': num_vars,
    'names': variables,
    'bounds': [[0.03e7, 0.15e7],
               [-27000, -15000],
               [3, 9],
               [3000, 300000]
               ]
}

print('Problem: ', problem)

N_fact = 10
N_skip = 15

param_values = sample(problem, 2**N_fact, skip_values=2**N_skip) # generate sample

print(f'Sample: factor {N_fact} with skipping factor {N_skip}')

def Run_Model(params_soil, params_plant):
    return HydraulicModel(E, h_b, params_soil, params_plant)


#  --------- Run model on sample  --------- 
for soil in soil_types_interested:
    logging.info(f"{soil}")

    # soil properties
    h0 = soil_Data[soil]['h0']
    k0 = soil_Data[soil]['k0']
    tau = soil_Data[soil]['tau']

    params_soil = [r0, h0, k0, tau]

    if __name__ == '__main__':

        with ProcessPoolExecutor() as executor:

            futures = executor.map(Run_Model, params_soil, param_values)

            futures = [executor.submit(Run_Model, params_soil, param) for param in param_values]
            results = [f.result().tolist() for f in futures]

        with open('2_Data/SOL_SA_{}.json'.format(soil), 'w') as f:
            json.dump(results, f, indent=4)

print('done')