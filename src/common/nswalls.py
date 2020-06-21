import numpy as np
from domain.dmplex import DMPlexDom

class NoSlipWalls:
    def __init__(self, lower, upper, sides=["left", "right", "up", "down"]):
        dim = len(lower)
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
        # if dim == 3:
        #     wallNaming.extend(["front", "back"])
        #     wallNumbering.extend([4,5])
        assert dim == 2
        self.walls = dict()

        for num, side in enumerate(sides):
            if side == "left":
                vertexs = self.left()
            elif side == "right":
                vertexs = self.right()
            elif side == "up":
                vertexs = self.up()
            elif side == "down":
                vertexs = self.down()
            else:
                raise Exception("Unknown side")
            self.walls[side] = Wall(num=num, vertexs=vertexs)
            self.walls[side].setWallName(side)

        self.wallsWithVelocity = list()
        self.staticWalls = ["left", "right", "up", "down"]
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
            print("side not defined")

    def getWallVelocity(self, name):
        try:
            wall = self.walls[name]
            wallVel = wall.getWallVelocity()
            wallVelDofs = wall.getVelDofs()
            if wallVel == 0:
                return [0] * len(self.lower)
            return wallVel, wallVelDofs
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
        return vertexs

    def right(self):
        vertexs = list()
        x_constant = self.upper[0]
        vertexs.append([x_constant, self.lower[1]])
        vertexs.append([x_constant, self.upper[1]])
        return vertexs

    def up(self):
        vertexs = list()
        y_constant = self.upper[1]
        vertexs.append([self.lower[0], y_constant])
        vertexs.append([self.upper[0], y_constant])
        return vertexs

    def down(self):
        vertexs = list()
        y_constant = self.lower[1]
        vertexs.append([self.lower[0], y_constant])
        vertexs.append([self.upper[0], y_constant])
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
        norm = self.computeNormal()
        velDofs = list(range(self.dim))
        velDofs.pop(norm)
        vel.pop(norm)
        cleanVel = list()
        cleanDofs = list()
        for dof, v in enumerate(vel):
            if(v != 0):
                cleanVel.append(v)
                cleanDofs.append(velDofs[dof])
        self.velocity = np.array(cleanVel)
        self.velDofs = cleanDofs

    def setWallName(self, name):
        self.name = name

    def getWallName(self):
        return self.name

    def getWallVelocity(self):
        try:
            vel = self.velocity
            return vel
        except: 
            return []

    def getVelDofs(self):
        try:
            return self.velDofs
        except:
            return []

    def getWallNum(self):
        return self.num

    def view(self):
        print(f"\nNo-Slip Wall Side {self.name} defined by {self.totalVecs} vector(s)")
        try:
            directions = ["X", "Y", "Z"]
            directions = [ directions[dof] for dof in self.getVelDofs() ]
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
    print(dofVelToSet)
    # print(ns)
    # plex.getFaceEntities("left")