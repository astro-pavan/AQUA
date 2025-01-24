import numpy as np
import fmodpy
import ctypes

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
        ctypes.byref(fnk_t),
        ctypes.byref(unk_t),
        ctypes.byref(cv),
        ctypes.byref(chit),
        ctypes.byref(chir),
        ctypes.byref(pmbar),
        ctypes.byref(uspec),
    )

    specific_entropy = ((unk_t.value - fnk_t.value) / T) * (uspec.value / unk_t.value) * 1e-10

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

# Example usage
if __name__ == "__main__":
    rho = 1.0  # g/cc
    t = 300.0  # K
    results = h2o_fit(rho, t)
    print(results)


