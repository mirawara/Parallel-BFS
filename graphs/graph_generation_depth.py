import numpy as np
from random import *

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        f.close()
nodi=8000
levels=20
offset=50
level_size=nodi/levels;

visitato = np.zeros(nodi)

n_archi=0
file="./graphs/"+str(nodi)+".graph"
grafo=open(file,"w")
for i in range(0,nodi):
    inf = max((int)(i/level_size)*level_size - offset, 0);
    sup = min(((int)(i/level_size) + 1)*level_size-1 + offset, nodi-1);
    s=""
    k=randint(2,100)
    n_archi+=k
   # if(i==0):
    #    b=randint(1,nodi)
    #else:
     #   b=randint(0,i)
    #s=s+" "+str(b)
    L=[]
    #L.append(b)
    connesso = False
    visitato[i] = 1
    for h in range(0,k):
        while True:
            j=randint(inf, sup)
            if(j!=i and j not in L):
                L.append(j)
                s=s+" "+str(j)
                if (visitato[j] == 1):
                    connesso = True
                visitato[j] = 1
                break
    
    if (i == 0):
        connesso = True
    if (connesso == False):
        while True:
            j=randint(inf, sup)
            if(j!=i and j not in L and visitato[j] == 1):
                L.append(j)
                s=s+" "+str(j)
                connesso = True
                n_archi += 1
                break
            
    
    grafo.write(s+"\n")
grafo.close()
line_prepender(file,str(nodi)+" "+str(n_archi))