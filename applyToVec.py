from petsc4py import PETSc
dof=3
dim=3
n=2


def funtion1dof(coords):
    return coords[0]*coords[1]

def funtion2dof(coords):
    return coords[0],coords[0]

def funtion3dof(coords):
    number= coords[0]*coords[1]*coords[2]
    return number, number, number

class Interface():
    """ """ 
    def ApplyToVector(self,vec, nodes, coords, dof, funtion):
        for i,coord in enumerate(coords):
            value= funtion(coord)
            index = [nodes[i]*dof + pos for pos in range(dof)]
            vec.setValues(index, value, addv=None)

vec = PETSc.Vec().createMPI((n**dim)*dof)
nodes = [i for i in range(n**dim)]
coords = []
for x in range(0,n,1):
    for y in range(0,n,1):
        if dim==2:
            coords.append((x,y))
        elif dim==3:
            for z in range(0,n,1):
                coords.append((x,y,z))
print(coords)
inter = Interface()
inter.ApplyToVector(vec,nodes,coords, dof,funtion3dof)
print(vec.view())
            