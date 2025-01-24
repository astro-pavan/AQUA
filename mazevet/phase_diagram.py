import pandas as pd
import matplotlib.pyplot as plt

eos_table = pd.read_table('Tables/aqua_eos_rhot_v1_0.dat', sep=' ', header=0, skiprows=21,
                          names=['rho', 'T', 'P', 'ad_grad', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase'])

print(eos_table)