import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys

if __name__=='__main__':
    
    print len(sys.argv)
    
    global_array = np.zeros((9658, 600), dtype=float)

    for ii in range(1, len(sys.argv)):
        print sys.argv[ii]
        a1 = np.loadtxt(sys.argv[ii], delimiter = ',', dtype = str,usecols=range(600))
        arr1 = a1.astype(np.float)
        for i in range(arr1.shape[0]):
            for j in range(arr1.shape[1]):
                if ii == 3:
                    global_array[i][j] += arr1[i][j] * 2

    res = open("./results/fused.csv","w")
    for i in range(global_array.shape[0]):
        for j in range(global_array.shape[1]):
           res.write(str(global_array[i][j]))
           res.write(',')
        res.write('\n')
    res.close()

        

