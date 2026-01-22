import random

def cscso_optimize(params):
    # Simplified placeholder for CSCSO optimization
    best_params = params.copy()
    for k in best_params:
        best_params[k] = max(1, params[k] - random.randint(0, 2))
    return best_params