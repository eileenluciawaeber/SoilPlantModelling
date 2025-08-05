import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

import random

import logging

import json as json

from scipy.optimize import differential_evolution
from sklearn.metrics import root_mean_squared_error

from concurrent.futures import ProcessPoolExecutor

# --------- Inputs --------- 
dx, dy, dz = 100, 100, 0.3

h_b = -1*np.arange(10,15010+100,dx, dtype=int) #bulk soil water potential in cm heads
E = 3e-6*np.arange(0,3000+1,dz) #transpiration rate in cm3 s**-1

# Load soil data
with open('2_Data/soil_Data.json') as f:
    soil_Data = json.load(f)

soil_types_interested = ["Clay", "Silt Loam", "Sandy Loam"]
soil_Data = {soil: soil_Data[soil] for soil in soil_types_interested}

soil = "Clay"
data_SOL = np.array(soil_Data[soil]["traject"])

r0 = 0.05 # cm root radius in cm
V_root = np.pi*3000

# for parallel processing
nrounds = 10 #number of evaluation rounds to calculate mean and stdv
ncpus = 70 #number of available cpus

nworkers_rounds = 10 #number of workers for round evaluation
nworkers_de = ncpus // nworkers_rounds #number of workers for differential evolution 

results = []

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


#  --------- Model  --------- 
def monotone(curve): # to make sure that the h_leaf curve will be monotonically increasing
    curve_mono = np.minimum.accumulate(curve) # make the function monotone
    mask_mono = curve_mono == np.nanmin(curve_mono) # all values that were set to minimum

    curve_mono[mask_mono] = np.nan # set the values where curve was initially increasing to nan values

    return curve_mono

def Calc_hleaf(E, h_b, params_plant):
    R_root, h0_x, tau_x  = params_plant

    k0_x = 1/R_root

    h_root = h_b

    #dissipation in the root
    h_x = h_root - R_root*E     # Couvreur model; from E = K_root*(h_root - h_x); R_root is the root's resistance

    # dissipation in the xylem including cavitation
    c_x = -k0_x/(1-tau_x)*(complex(h0_x)**tau_x)
    h_leaf = -1*np.abs(E/c_x+h_x.astype(np.complex128)**(1-tau_x))**(1/(1-tau_x))

    return monotone(h_leaf)

def Calc_SOL(E, hb_vect, params_plant, gradmax = 0.7):
    SOL_hleaf = []
    SOL_E = []

    for _, hb in enumerate(hb_vect):
        h_leaf = Calc_hleaf(E, hb,params_plant) # Calculate the iso psi soil curve
        
        FX = np.gradient(E, -h_leaf)

        rel_grad = np.abs(FX)/np.nanmax(np.abs(FX))
        pos = np.nanmax(np.where(rel_grad >= gradmax)[0]) # Find earliest position where condition holds (where gradient is larger than threshold)
        
        # add the E and h_leaf values to the SOL
        SOL_E.append(E[pos])
        SOL_hleaf.append(h_leaf[pos])
    
    return np.column_stack([hb_vect, np.array(SOL_hleaf), np.array(SOL_E)])    


#  --------- Loss / Objective ---------
def calc_error(M_simple, M_complex):

    # Extract columns
    _, leaf_wp_s, E_s = M_simple.T
    _, leaf_wp_c, E_c = M_complex.T

    # Compute scaling factors for leaf and E vectors
    r_leaf = 0 - np.min(leaf_wp_c)
    r_E = np.max(E_c) - np.min(E_c)

    fact_leaf = 1 / r_leaf if r_leaf != 0 else 1
    fact_E = 1 / r_E if r_E != 0 else 1

    # Compare transpiration
    rmse_E = root_mean_squared_error(E_s, E_c)

    # Compare leaf WP
    rmse_leaf = root_mean_squared_error(leaf_wp_s, leaf_wp_c)

    return fact_E * rmse_E + fact_leaf * rmse_leaf

def objective(params, E, h_b, gradmax = 0.7):

    logging.info(f"Evaluating params: {params}")

    SOL = Calc_SOL(E, h_b, params_plant=params, gradmax=gradmax)

    error = calc_error(SOL, data_SOL) # calculate error

    if np.isnan(error) or np.isinf(error):
        logging.info(f"Error is nan or inf with {params}")
        return 1e6

    return error


# --------- Optimization Function ---------
def run_de():
    # Random seed for each iteration / function call
    seed = random.randint(0, 1e6)

    logging.info(f"Seed: {seed}")

    # Bounds to ensure realistic values
    bounds = [(0.03e7, 0.15e7), (-27000, -15000), (3, 9)]  

    # Sample E and h_b inputs
    args_sample = (E, h_b)

    niter = 50
    popsize = 12

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=args_sample,
        strategy='best2bin',
        maxiter=niter,
        popsize=popsize,
        tol=0.3,
        seed=seed,
        workers=nworkers_de,
        updating='deferred'
    )

    # Optimized parameters
    optimized_R_root, optimized_h0_x, optimized_tau_x = result.x

    logging.info(f"DE Results: {optimized_R_root, optimized_h0_x, optimized_tau_x}")

    return [optimized_R_root, optimized_h0_x, optimized_tau_x]

def run_de_wrapper(_):
    return run_de()


# --------- Multiprocessing ---------
if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=nworkers_rounds) as executor:
        futures = executor.map(run_de_wrapper, range(nrounds))

        for i, res in enumerate(futures):
            results.append(res)

    res_R_root = list(zip(*results))[0]
    res_h0_x = list(zip(*results))[1]
    res_tau_x = list(zip(*results))[2]

    print("R_root:", res_R_root)
    print("h0_x:", res_h0_x)
    print("tau:", res_tau_x)