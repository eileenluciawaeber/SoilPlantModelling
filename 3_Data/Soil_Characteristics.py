import numpy as np


# ----- Initialize the soil data -----
xx=np.array([   [0.09, 0.475, 0.0268, 37.31, 0.131, 1.44, (0+45)/2, (45-0)/2], #clay
                [0.075, 0.366, 0.0386, 25.91, 0.194, 5.52, (20+45)/2, (45-20)/2], #clay loam
                [0.027, 0.463, 0.0897, 11.15, 0.22, 31.68, (23+52)/2, (52-23)/2], #loam
                [0.035,	0.437,	0.115,	8.70,	0.474,	146.64, (70+86)/2, (86-70)/2], #loamy sand
                [0.02, 0.437, 0.138, 7.25, 0.592, 504, (86+100)/2, (100-86)/2], #sand
                [0.068, 0.398, 0.0356, 28.09, 0.25, 10.32, (45+80)/2, (80-45)/2], #sandy clay loam
                [0.041,	0.453,	0.0682,	14.66,	0.322,	62.16,   (50+70)/2, (70-50)/2],  #sandy loam
                [0.015, 0.501, 0.0482, 20.75, 0.211, 16.32, (20+50)/2, (50-20)/2],  #silt loam
                [0.04, 0.471, 0.0307, 32.57, 0.151, 3.6, (0+20)/2, (20-0)/2] #silty clay loam
    ])

nb_soils = len(xx)

soil_types = ["Clay", "Clay Loam", "Loam", "Loamy Sand", "Sand", "Sandy Clay Loam", "Sandy Loam", "Silt Loam", "Silty Clay Loam"]


thr_xx = xx[:,0]
ths_xx = xx[:,1]
h0_xx  = -1*xx[:,3]
l_xx   = xx[:,4]
k0_xx  = xx[:,5] / (24*60*60) # cm/s - for the blue 0.2; 3*10^-3
tau_xx = 2+3*l_xx #corresponding to 2+l*(a+2) with l exp for theta(h) and a tortuosity

hxx = np.array([thr_xx, ths_xx, h0_xx, l_xx, k0_xx, tau_xx])


# ----- Create dictionary including the soil data -----
soil_Data = {}

soil_Data = {
    soil: {
        "thr": thr_xx[i],
        "ths": ths_xx[i],
        "h0": h0_xx[i],
        "k0": k0_xx[i],
        "tau": tau_xx[i]
    } 
    for i, soil in enumerate(soil_types)
}