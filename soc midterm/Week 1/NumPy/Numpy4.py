import numpy as np
bef=np.array([[1,2,3],[4,5,6]])
print(bef)
aft=bef.reshape([3,2])
print(aft)
#used for vertical stacking
v1=np.array([1,2,3])
v2=np.array([4,5,6])
print(np.vstack([v1,v2,v2,v2]))
#similarly we can  use np.hstack for horizontal stacking also.