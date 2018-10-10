from MxNx1_Neural_Network_Analysis import my_histogram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x1=np.random.normal(loc=0,scale=1, size=(30,1))
x2=np.random.normal(loc=0,scale=1, size=(30,1))
X=np.concatenate((x1,x2),axis=1)


print (x1)
