#load data from file
import numpy as np
file=np.genfromtxt('data1.txt' , delimiter=',' )
print(file.astype('int32'))#for printing in int format
print(file>50)
print(file[file>50])