import matplotlib.pyplot as plt
labels=['A','B','C']
Values=[1,4,2]
bars=plt.bar(labels,Values)
#bars[0].set_hatch('/')
#bars[1].set_hatch('o')
#bars[2].set_hatch('*')
patterns=['/','o','*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()