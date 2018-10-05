import random
import string
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 

def rand_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def editDistance(x,y):
    if(len(x)) == 0:
        return len(y)
    if(len(y)) == 0:
        return len(x)
    if(x[-1] == y[-1]):
        return editDistance(x[:-1],y[:-1])
    return 1+ min(editDistance(x[:-1],y[:-1]),editDistance(x,y[:-1]),editDistance(x[:-1],y))

def editDistance_memo(x,y,states=None):
    if states is None:
        states = np.full((len(x), len(y)), -1)
    if(len(x)) == 0:
        return len(y)
    if(len(y)) == 0:
        return len(x)
    if(states[len(x)-1,len(y)-1] != -1):
        return states[len(x)-1,len(y)-1]
    if(x[-1] == y[-1]):
        return editDistance_memo(x[:-1],y[:-1])
    states[len(x)-1,len(y)-1] = 1+ min(editDistance_memo(x[:-1],y[:-1],states),editDistance_memo(x,y[:-1],states),editDistance_memo(x[:-1],y,states))
    return states[len(x)-1,len(y)-1]

def editDistance_iterative(x,y):
    states = np.zeros((len(x)+1,len(y)+1))
    states[0, 1:] = range(1,len(y)+1)
    states[1:, 0] = range(1,len(x)+1)
    for i in range(1,len(x)+1):
        for j in range(1,len(y)+1):
            if(x[i-1]!= y[j-1]):
                states[i,j] = min(states[i-1,j]+1,states[i,j-1]+1,states[i-1,j-1]+1)
            else:
                states[i,j] = min(states[i-1,j]+1,states[i,j-1]+1,states[i-1,j-1])
    return(states[len(x),len(y)])

results_recursion = []
results_memo = []
results_iterative = []

for i in range(1,12):
    print("Iteration " + str(i))
    rand_string1 = rand_string(i)
    rand_string2 = rand_string(i)
    start=datetime.now()
    editDistance(rand_string1,rand_string2)
    results_recursion.append((datetime.now()-start).total_seconds())


for i in range(1,40):
    print("Iteration " + str(i))
    rand_string1 = rand_string(i)
    rand_string2 = rand_string(i)
    start=datetime.now()
    editDistance_memo(rand_string1,rand_string2)
    results_memo.append((datetime.now()-start).total_seconds())


for i in range(1,200):
    print("Iteration " + str(i))
    rand_string1 = rand_string(i)
    rand_string2 = rand_string(i)
    start=datetime.now()
    editDistance_iterative(rand_string1,rand_string2)
    results_iterative.append((datetime.now()-start).total_seconds())


plt.plot(results_recursion)
plt.title("Vrijeme izvrsavanja rekurzivnog algoritma")
plt.ylabel('Broj sekundi')
plt.xlabel('Duljina znakovnog niza')
plt.show()

plt.plot(results_memo)
plt.title("Vrijeme izvrsavanja rekurzivnog algoritma s memoriranjem")
plt.ylabel('Broj sekundi')
plt.xlabel('Duljina znakovnog niza')
plt.show()

plt.plot(results_iterative)
plt.title("Vrijeme izvrsavanja iterativnog algoritma")
plt.ylabel('Broj sekundi')
plt.xlabel('Duljina znakovnog niza')
plt.show()

