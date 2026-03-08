import numpy as np
try:
    a = np.array([1, 2, 3])
    a.shape[1]
except Exception as e:
    print("a.shape[1] 1D:", type(e), e)

try:
    a = np.array(1)
    a.shape[1]
except Exception as e:
    print("a.shape[1] 0D:", type(e), e)

try:
    [1][1]
except Exception as e:
    print("list[1]:", type(e), e)

try:
    np.array([1])[1]
except Exception as e:
    print("np.array[1]:", type(e), e)
