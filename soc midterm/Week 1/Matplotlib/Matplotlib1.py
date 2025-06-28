import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.figure(figsize=(5,3),dpi=300)
x=[0,1,2,3]
y=[0,1,2,3]
x2=np.arange(0,4.5,0.5)
plt.plot(x2,x2**2)
plt.title('First Graph' , fontdict= {'fontname': 'Comic Sans MS' , 'fontsize' :20})
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.xticks([1,2,3])
plt.yticks([2,4,6,8,10])
plt.plot(x,y,label='y=x',color='red',linewidth=2,marker='*',markersize=10,markeredgecolor='black',linestyle=':')
plt.legend()
plt.savefig('mygraph.png',dpi=300)
plt.show()