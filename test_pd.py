import pandas as pd
import numpy as np

df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
try:
    df[:, 0]
except Exception as e:
    print("df slice:", type(e), e)

shape_test = (10,)
try:
    shape_test[1]
except Exception as e:
    print("tuple slice:", type(e), e)
