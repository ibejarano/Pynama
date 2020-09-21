import numpy as np
from domain.dmplex import DMPlexDom
import sys

class NoSlipWalls:
    def __init__(self, lower, upper, exclude=[] ):
        dim = len(lower)
        self.dim = dim

        self.lower = lower
        self.upper = upper
        #           up
        #        ________
        #       |        |
        #left   |        | right
        #       |________|
        #  -->x    down
        #
        # wallNumbering = [0, 1, 2, 3]
        self.walls = dict()

        if self.dim == 2:
            sides=[ "left", "right", "up", "down" ]
        else:
            sides=[ "left", "right", "up", "down", "back", "front" ]

        for num, side in enumerate(sides):
            if not side in exclude:
                if side == "left":
                    vertexs = self.left()
                elif side == "right":
                    vertexs = self.right()
                elif side == "up":
                    vertexs = self.up()
                elif side == "down":
                    vertexs = self.down()
                elif side == "front":
                    vertexs = self.front()
                elif side == "back":
                    vertexs = self.back()
                else:
                    raise Exception("Unknown side")
                self.walls[side] = Wall(num=num, vertexs=vertexs)
                self.walls[side].setWallName(side)

        self.staticWalls = list(self.walls.keys())
        self.wallsWithVelocity = list()
        self.computeWallsNormals()

    def __iter__(self):
        for side in self.walls.keys():
            yield self.walls[side]

    def __len__(self):
        return len(self.walls.keys())

    def __repr__(self):
        for side in self:
            side.view()
        return f"Walls defined: {len(self)} "

    def getWallsNames(self):
        return self.walls.keys()

    def getWallsWithVelocity(self):
        return self.wallsWithVelocity

    def getWallBySideName(self, name):
        return self.walls[name]

    def getStaticWalls(self):
        return self.staticWalls

    def setWallVelocity(self, name, vel):
        try:
            dim = len(self.lower)
            assert dim == len(vel)
            wall = self.walls[name]
            wall.setWallVelocity(vel)
            self.wallsWithVelocity.append(name)
            self.staticWalls.remove(name)
        except:
            return None

    def getWallVelocity(self, name):
        try:
            wall = self.walls[name]
            return wall.getWallVelocity()
        except:
            print("side not defined")

    def getStaticDofsByName(self, name):
        try:
            wall = self.walls[name]
            return wall.getStaticDofs()
        except:
            print("side not defined")

    def computeWallsNormals(self):
        normals = dict()
        for wall in self:
            name = wall.getWallName()
            nsNormal = wall.computeNormal()
            normals[name] = nsNormal
        self.normals = normals

    def getWalletNormalBySideName(self, name):
        try:
            nsNormal = self.normals[name]
            return nsNormal
        except:
            print("side not defined")

    def left(self):
        vertexs = list()
        x_constant = self.lower[0]
        vertexs.append([x_constant, self.lower[1]])
        vertexs.append([x_constant, self.upper[1]])
        if self.dim == 3:
            vertexs[0].append(0)
            vertexs[1].append(0)
        return vertexs

    def right(self):
        vertexs = list()
        x_constant = self.upper[0]
        vertexs.append([x_constant, self.lower[1]])
        vertexs.append([x_constant, self.upper[1]])
        if self.dim == 3:
            vertexs[0].append(0)
            vertexs[1].append(0)
        return vertexs

    def up(self):
        vertexs = list()
        y_constant = self.upper[1]
        vertexs.append([self.lower[0], y_constant])
        vertexs.append([self.upper[0], y_constant])
        if self.dim == 3:
            vertexs[0].append(0)
            vertexs[1].append(0)
        return vertexs

    def down(self):
        vertexs = list()
        y_constant = self.lower[1]
        vertexs.append([self.lower[0], y_constant])
        vertexs.append([self.upper[0], y_constant])
        if self.dim == 3:
            vertexs[0].append(0)
            vertexs[1].append(0)
        return vertexs

    def front(self):
        vertexs = list()
        z_constant = self.upper[2]
        vertexs.append([self.lower[0], z_constant, 0])
        vertexs.append([self.upper[0], z_constant, 0])
        return vertexs
    
    def back(self):
        vertexs = list()
        z_constant = self.lower[2]
        vertexs.append([self.lower[0], z_constant, 0])
        vertexs.append([self.upper[0], z_constant, 0])
        return vertexs

class Wall:
    def __init__(self, num, vertexs=None):
        self.totalVecs = len(vertexs) - 1
        self.num = num
        assert self.totalVecs > 0
        self.dim = len(vertexs[0])
        node = Vertex(pointCoords=vertexs.pop(0))
        self.head = node
        for vertex in vertexs:
            node.next = Vertex(pointCoords=vertex)
            node = node.next

        normal = self.computeNormal()
        velDofs = list(range(self.dim))
        velDofs.pop(normal)
        self.staticDofs = velDofs   

    def __repr__(self):
        node = self.head
        nodes = list()
        while node is not None:
            nodes.append(str(node.data))
            node = node.next
        nodes.append("None")
        print(f"wall name: {self.name}")
        return " -> ".join(nodes)

    def __iter__(self):
        node = self.head
        while node is not None and node.next is not None :
            yield node , node.next
            node = node.next

    def setWallVelocity(self, vel):
        velDofs = list()
        vels = list()
        staticDofs = list(self.staticDofs)
        for dof in staticDofs:
            if vel[dof] != 0:
                vels.append(vel[dof])
                velDofs.append(dof)
                rmDof = self.staticDofs.index(dof)
                self.staticDofs.pop(rmDof)
        if len(velDofs) > 0:
            self.velocity = np.array(vels)
            self.velDofs = velDofs
        else:
            raise Exception("Velocity not valid")

    def setWallName(self, name):
        self.name = name

    def getWallName(self):
        return self.name

    def getWallVelocity(self):
        try:
            return self.velocity, self.velDofs
        except:
            vel = [0] * len(self.staticDofs)
            return vel, self.staticDofs

    def getStaticDofs(self):
        return self.staticDofs

    def getWallNum(self):
        return self.num

    def view(self):
        print(f"\nNo-Slip Wall Side {self.name} defined by {self.totalVecs} vector(s)")
        try:
            directions = ["X", "Y", "Z"]
            directions = [ directions[dof] for dof in self.getWallVelocity()[1] ]
            assert len(self.getWallVelocity() > 0)
            print(f"Wall Velocity {self.getWallVelocity()} in {directions} direction(s) ")
        except:
            print(f"Static Wall")
        for vecNum, vec in enumerate(self):
            norm = self.computeNormal()
            print(f" vec {vecNum} : from {vec[0]} to {vec[1]} , normal: {norm}")

    def computeNormal(self):
        """Return a number representing the normal
        0: x normal
        1: y normal
        2: z normal
        """
        # TODO Valid for 2-D only!
        if self.num in [0, 1, 2, 3]:
            z_direction = [ 0, 0, 1]
            for vertex in self:
                vectorTail = vertex[0].getCoordinates() 
                vectorHead = vertex[1].getCoordinates()
                vec = np.abs(vectorHead - vectorTail)
                vec = vec / np.linalg.norm(vec)
                vec = np.cross(vec, z_direction)
                vec = list(np.abs(vec))
                norm = vec.index(1.0)
            return norm
        else:
            return 2

class Vertex:
    def __init__(self, pointCoords):
        self.data = pointCoords
        self.next = None

    def __repr__(self):
        return str(self.data)

    def getCoordinates(self):
        return np.array(self.data)


if __name__ == "__main__":
    lower=[0,0]
    upper=[1,1]
    plex = DMPlexDom(lower=lower, upper=upper , faces=[3,3] )
    ns = NoSlipWalls(lower, upper)
    ns.setWallVelocity("left", [3,5])
    ns.setWallVelocity("up", [34,12])
    vel, velDofs = ns.getWallVelocity("up")
    dim = 2
    nodesSet = [3, 5 , 9]
    dofVelToSet = [node*dim + dof for node in nodesSet for dof in velDofs]
    # print(ns)
    # plex.getFaceEntities("left")