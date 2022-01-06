import numpy as np
import matplotlib.pyplot as plt



x0=[9/10,9/11,10/11,10/13,11/12,11/13,11/14,11/15,
    11/20,12/13,12/17,13/15,13/16,13/18,14/15,14/17,
    14/19,15/19,16/17,16/19,17/19,17/20,19/20]


x2=[1/4,1/5,1/7,2/5,2/11,3/8,4/5,4/7,
    5/6,5/7,5/8,5/11,5/13,6/7,6/11,7/11,
    7/13,8/9,9/13,9/14,9/20,10/17,11/17,12/19,
    17/18]


x1=[1/1,2/1,3/1,6/1,8/1,9/1,10/1,11/1,
    12/1,13/1,14/1,15/1,16/1,17/1,18/1,19/1,
    20/1,3/2,7/2,9/2,13/2,15/2,17/2,19/2,
    3/4,3/5,7/3,10/3,11/3,13/3,14/3,16/3,
    17/3,19/3,20/3,9/4,11/4,13/4,15/4,17/4,
    19/4,9/5,12/5,14/5,16/5,17/5,18/5,19/5,
    13/6,17/6,19/6,8/7,9/7,10/7,12/7,15/7,
    16/7,17/7,18/7,19/7,20/7,8/11,13/8,15/8,
    17/8,19/8,17/5,17/9,19/9,19/10,18/11,19/11,
    13/14,17/13,13/19,13/20,15/16,15/17,18/19,16/11]

x1Inv=[1/elem for elem in x1]
x2Inv=[1/elem for elem in x2]
x0Inv=[1/elem for elem in x0]
fig,ax=plt.subplots(figsize=(20,20))
ax.scatter(x0,[0]*len(x0),color="red")
ax.scatter(x0Inv,[0]*len(x0Inv),color="red")
ax.scatter(x1,[0]*len(x1),color="blue")
ax.scatter(x1Inv,[0]*len(x1Inv),color="brown")
ax.scatter(x2,[0]*len(x2),color="green")
ax.scatter(x2Inv,[0]*len(x2Inv),color="green")
ax.set_xscale("log")
plt.savefig("fraction.png")
print(min(x2))
print(max(x2))
print(min(x2Inv))
print(max(x2Inv))