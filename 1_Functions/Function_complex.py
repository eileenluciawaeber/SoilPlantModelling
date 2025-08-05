import numpy as np


# ----- Calculation of hleaf for each soil WP -----
def monotone(curve): # to make sure that the h_leaf curve will be monotonically increasing
    curve_mono = np.minimum.accumulate(curve) # make the function monotone
    mask_mono = curve_mono == np.nanmin(curve_mono) # all values that were set to minimum

    curve_mono[mask_mono] = np.nan # set the values where curve was initially increasing to nan values

    return curve_mono

def Calc_hleaf(E, h_b, params_soil, params_plant):
    r0, h0, k0, tau = params_soil
    R_root, h0_x, tau_x, L  = params_plant

    V_root = np.pi*3000
    
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


# ----- Calculate the SOL -----
def Calc_SOL(E, hb_vect, params_soil, params_plant, gradmax = 0.7):
    SOL_hleaf = []
    SOL_E = []

    for i, hb in enumerate(hb_vect):
        h_leaf = Calc_hleaf(E, hb, params_soil, params_plant) # Calculate the iso psi soil curve
        
        FX = np.gradient(E, -h_leaf)

        if np.sum(np.isnan(FX)) == len(FX):
            hb_vect_red = hb_vect[:i]
            return np.column_stack([hb_vect_red, np.array(SOL_hleaf), np.array(SOL_E)])

        rel_grad = np.abs(FX)/np.nanmax(np.abs(FX))
        pos = np.nanmax(np.where(rel_grad >= gradmax)[0]) # Find earliest position where condition holds (where gradient is larger than threshold)
        
        # add the E and h_leaf values to the SOL
        SOL_hleaf.append(h_leaf[pos])
        SOL_E.append(E[pos])
    
    return np.column_stack([hb_vect, np.array(SOL_hleaf), np.array(SOL_E)])