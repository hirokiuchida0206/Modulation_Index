import MI
import numpy as np

MI.MI_calculation(np.random.rand(2000)-0.5, 400, [7, 9], [120, 130])
#MI.freq_response_bandpass(534, [7, 15], 1000)
# tap = MI.tap_calc(1000, 5)
# print(tap)