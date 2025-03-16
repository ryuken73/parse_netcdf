import numpy as np
in_file = 'ir105_conversion.txt'
out_file = 'ir105_conversion_c.txt'
kArry = np.loadtxt(in_file)
cArry = kArry - 273
np.savetxt(out_file, cArry, fmt='%.4f')