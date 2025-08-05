import numpy as np

from sklearn.metrics import root_mean_squared_error


# ----- Define the weughted and normalized RMSE -----
def calc_error(M_simple, M_complex):

    # Extract columns
    _, leaf_wp_s, E_s = M_simple.T
    _, leaf_wp_c, E_c = M_complex.T

    # Compute scaling factors for leaf and E vectors
    r_leaf = 0 - np.min(leaf_wp_c) # going to zero to include well-watered transpiration
    r_E = np.max(E_c) - np.min(E_c)

    fact_leaf = 1 / r_leaf if r_leaf != 0 else 1
    fact_E = 1 / r_E if r_E != 0 else 1

    # Compare transpiration
    rmse_E = root_mean_squared_error(E_s, E_c)

    # Compare leaf WP
    rmse_leaf = root_mean_squared_error(leaf_wp_s, leaf_wp_c)

    return fact_E * rmse_E + fact_leaf * rmse_leaf