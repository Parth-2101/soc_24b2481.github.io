import numpy as np
a=np.array([[1,2,3,4,5,6],[5,8,2,3,6,9]])
print(a)
print(a[1,3])
print(a[1:])
print(a[0, 1:-1:2])#startindex, range:stepSize
a[1,5]=10
a[:,2]=[1,2]
print(a)
b=np.full_like(a,4)
print(b)