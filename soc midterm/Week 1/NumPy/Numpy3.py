import numpy as np
a=np.array([1,2,3])
b=a.copy()
b[0]=100
print(a)#a is unchanged due to copy() f^n
print(b)
c=3
print(a*c)#list*int is possible in numpy
print(np.sin(a))