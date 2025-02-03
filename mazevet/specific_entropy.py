import numpy as np
import fmodpy
import ctypes
from scipy.interpolate import interp1d, RegularGridInterpolator

#mazevet21 = fmodpy.fimport('mazevet/eoswater21.f')

# Load the shared library
lib = ctypes.CDLL("./mazevet/libeoswater21.so")

# Define the Fortran subroutine signature
# All arguments are passed by reference in Fortran
lib.h2ofit_.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # RHO
    ctypes.POINTER(ctypes.c_double),  # T
    ctypes.POINTER(ctypes.c_double),  # PnkT
    ctypes.POINTER(ctypes.c_double),  # FNkT
    ctypes.POINTER(ctypes.c_double),  # UNkT
    ctypes.POINTER(ctypes.c_double),  # CV
    ctypes.POINTER(ctypes.c_double),  # CHIT
    ctypes.POINTER(ctypes.c_double),  # CHIR
    ctypes.POINTER(ctypes.c_double),  # PMbar
    ctypes.POINTER(ctypes.c_double),  # USPEC
]

# Python wrapper function
def h2o_fit(rho, T):
    # Inputs
    rho_c = ctypes.c_double(rho)
    T_c = ctypes.c_double(T)
    
    # Outputs
    pnk_t = ctypes.c_double()
    fnk_t = ctypes.c_double()
    unk_t = ctypes.c_double()
    cv = ctypes.c_double()
    chit = ctypes.c_double()
    chir = ctypes.c_double()
    pmbar = ctypes.c_double()
    uspec = ctypes.c_double()

    # Call the Fortran subroutine
    lib.h2ofit_(
        ctypes.byref(rho_c),
        ctypes.byref(T_c),
        ctypes.byref(pnk_t), 
        ctypes.byref(fnk_t), # dimensionless
        ctypes.byref(unk_t), # dimensionless
        ctypes.byref(cv),
        ctypes.byref(chit),
        ctypes.byref(chir),
        ctypes.byref(pmbar),
        ctypes.byref(uspec), # in erg/g
    )

    u = uspec.value * 1e-4

    specific_entropy = ((unk_t.value - fnk_t.value) / T) * (u / unk_t.value)

    # Return results as a dictionary
    # return {
    #     "PnkT": pnk_t.value,
    #     "FNkT": fnk_t.value,
    #     "UNkT": unk_t.value,
    #     "CV": cv.value,
    #     "CHIT": chit.value,
    #     "CHIR": chir.value,
    #     "PMbar": pmbar.value,
    #     "USPEC": uspec.value,
    # }

    return specific_entropy

def is_in_mazevet(P, T):
    # Define boundaries
    T_boundary_3_7 = np.linspace(300, 2250)
    P_boundary_3_7 = 700e9 * np.ones_like(T_boundary_3_7)
    
    T_boundary_5_7 = np.linspace(2250, 4000)
    P_boundary_5_7 = 10 ** (np.log10(42e9) - np.log10(6) * (((T_boundary_5_7 / 1000) - 2) / 18))
    
    T_boundary_6_7 = np.linspace(4000, 30000)
    P_boundary_6_7 = 0.05e9 + (3e9 - 0.05e9) * (((T_boundary_6_7 / 1000) - 1) / 39)
    
    P_boundary_5_7_isothermal = np.linspace(P_boundary_5_7[-1], P_boundary_6_7[0])
    T_boundary_5_7_isothermal = 4000 * np.ones_like(P_boundary_5_7_isothermal)
    
    P_boundary_3_7_isothermal = np.linspace(P_boundary_3_7[-1], P_boundary_5_7[0])
    T_boundary_3_7_isothermal = 2250 * np.ones_like(P_boundary_5_7_isothermal)
    
    # Concatenate all boundary points
    T_boundaries = np.concatenate([T_boundary_3_7, T_boundary_3_7_isothermal, 
                                   T_boundary_5_7, T_boundary_5_7_isothermal, T_boundary_6_7])
    P_boundaries = np.concatenate([P_boundary_3_7, P_boundary_3_7_isothermal, 
                                   P_boundary_5_7, P_boundary_5_7_isothermal, P_boundary_6_7])
    
    P_inner_boundaries = P_boundaries * 2
    
    # Sort boundary points in ascending order of T
    sorted_indices = np.argsort(T_boundaries)
    T_boundaries = T_boundaries[sorted_indices]
    P_boundaries = P_boundaries[sorted_indices]
    P_inner_boundaries = P_inner_boundaries[sorted_indices]
    
    # Interpolate boundary curve
    P_interp = interp1d(T_boundaries, P_boundaries, bounds_error=False, fill_value=(P_boundaries[0], P_boundaries[-1]))
    P_inner_interp = interp1d(T_boundaries, P_inner_boundaries, bounds_error=False, fill_value=(P_boundaries[0], P_boundaries[-1]))
    
    # Check if (T, P) is inside the region
    return P >= P_interp(T), P >= P_inner_interp(T)

# Example usage
if __name__ == "__main__":
    
    filename = './SESAME_table/AQUA_H20.txt'

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract metadata
    metadata = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            metadata.append(line)
        else:
            break

    # Extract numerical data
    data_start_idx = len(metadata) + 1
    date = int(lines[data_start_idx - 1])
    num_values = [int(v) for v in lines[data_start_idx].split()]
    num_rho, num_T = num_values[0], num_values[1]

    rho = np.array([float(v) for v in lines[data_start_idx + 1].split()])
    T = np.array([float(v) for v in lines[data_start_idx + 2].split()])

    table = np.loadtxt(lines[data_start_idx + 3:], dtype=np.float64)
    u, P, c, s = table[:, 0], table[:, 1], table[:, 2], table[:, 3]

    u = u.reshape(num_rho, num_T)
    P = P.reshape(num_rho, num_T)
    c = c.reshape(num_rho, num_T)
    s = s.reshape(num_rho, num_T)

    A2_T, A2_rho, = np.meshgrid(T, rho)

    assert A2_T.shape == P.shape

    replace_mask, inner_mask = is_in_mazevet(P, A2_T)
    transition_mask = replace_mask & ~inner_mask

    for i_rho, rho_val in enumerate(rho):
        for i_T, T_val in enumerate(T):
            P_val = P[i_rho, i_T]
            if replace_mask[i_rho, i_T]:
                s_old = s[i_rho, i_T]
                s_new = h2o_fit(rho_val, T_val)
                s[i_rho, i_T] = s_new
                print(f'(rho: {rho_val:.1e}, P: {P_val:.1e}, T: {T_val:.1e}) : {s_old:.3e} --> {s_new:.3e}')

    modified_file = './SESAME_table/AQUA_H20_v2.txt'

    with open(modified_file, 'w') as f:
        f.writelines(metadata)
        f.write(f'{date}\n')
        f.write(f"{num_rho} {num_T}\n")
        f.write(" ".join(f"{v:.8e}" for v in rho) + "\n")
        f.write(" ".join(f"{v:.8e}" for v in T) + "\n")
        
        for i in range(num_rho):
            for j in range(num_T):
                f.write(f"{u[i, j]:.8e} {P[i, j]:.8e} {c[i, j]:.8e} {s[i, j]:.8e}\n")


    print('DONE')