from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

class XmlGenerator(object):
    def __init__(self, dim, h5name):
        self.root = Element('Xdmf')
        self.root.set('Version', '2.0')
        self.dim = dim
        self.h5name = h5name

    def setUpDomainNodes(self, totalNodes=None, nodesPerDim=None):
        """
        totalNodes: int
        nodesPerDim: [ int ]
        """
        try:
            assert totalNodes != None
            self.dimensions = totalNodes
        except:
            assert len(nodesPerDim) == self.dim 
            self.dimensions = 1
            for nodes in nodesPerDim:
                self.dimensions *= nodes

    def generateXMLTemplate(self):
        self.domain = SubElement(self.root, 'Domain')
        self.grid = SubElement(self.domain, 'Grid')
        self.grid.set('Name',"TimeSeries")
        self.grid.set('GridType',"Collection")
        self.grid.set('CollectionType',"Temporal")

    def generateMeshData(self, name):
        meshGrid = SubElement(self.grid, 'Grid')
        meshGrid.set('Name', name)
        meshGrid.set('GridType', "uniform")
        topology = SubElement(meshGrid, "Topology")
        topology.set("TopologyType", "Polyvertex")
        topology.set("Dimensions", str(self.dimensions))
        geometry = SubElement(meshGrid, "Geometry")
        geometry.set("GeometryType", "XY" if self.dim == 2 else "XYZ")
        geometryData = SubElement(geometry, "DataItem")
        geometryData.set("Dimensions" , str(self.dimensions*self.dim))
        geometryData.set("NumberType", "Float")
        geometryData.set("Format","HDF")
        geometryData.text = "mesh.h5:/fields/mesh"

        return meshGrid

    def setTimeStamp(self, t, meshElem):
        timestamp = SubElement(meshElem, "Time")
        timestamp.set("Value", str(t))

    def setVectorAttribute(self, name, step, meshGrid):
        attr = SubElement(meshGrid, "Attribute")
        attr.set("Name", name)
        attr.set("AttributeType", "Vector")
        attr.set("Center", "Node")

        attrData = SubElement(attr, "DataItem")
        attrData.set("ItemType", "Function")
        attrData.set("Dimensions", "{} {}".format(self.dimensions, self.dim))
        joinFunction = self.getJoinString(self.dim)
        attrData.set("Function", joinFunction)
        for i in range(self.dim):
            self.setDataToAttribute(attrData, step, name, i)

    def setScalarAttribute(self, name, step, meshGrid):
        attr = SubElement(meshGrid, "Attribute")
        attr.set("Name", name)
        attr.set("AttributeType", "Scalar")
        attr.set("Center", "Node")

        attrData = SubElement(attr, "DataItem")
        attrData.set("Dimensions", "{}".format(self.dimensions))
        attrData.set("NumberType", "Float")
        attrData.set("Format", "HDF")
        attrData.text = f"{self.h5name}-{step:05d}.h5:/fields/{name}"

    def setDataToAttribute(self, attrData, step, name, dof):
        dofs = ['X', 'Y', 'Z']
        hyperSlab = SubElement(attrData, "DataItem")
        hyperSlab.set("ItemType", "HyperSlab")
        hyperSlab.set("Dimensions", str(self.dimensions))
        hyperSlab.set("Name", "{}-{}".format(name, dofs[dof]))

        dimData = SubElement(hyperSlab, "DataItem")
        # TODO: Esto en 3 1 tira error, no se porque pero para dim 2 es asi
        dimData.set("Dimensions", "3 1")
        # dimData.set("Dimensions", "{} 1".format(self.dim))
        dimData.set("Format","XML")
        dimData.text = "{} {} {}".format(dof, self.dim, self.dimensions)

        velData = SubElement(hyperSlab, "DataItem")
        velData.set("Dimensions", str(self.dimensions * self.dim))
        velData.set("NumberType", "Float")
        velData.set("Format", "HDF")
        velData.text = f"{self.h5name}-{step:05d}.h5:/fields/{name}"

    def writeFile(self, nameFile):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        f = open("{}.xmf".format(nameFile), "w")
        prett = reparsed.toprettyxml(indent=" ")
        f.write(prett)
        f.close()

    def printify(self):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        print(reparsed.toprettyxml(indent=" "))

    @staticmethod
    def formatStep(step):
        maxZeros = 5
        step = str(step)
        zerosToAdd = maxZeros - len(step)
        zerosInFront = '0' * zerosToAdd
        return zerosInFront + step

    @staticmethod
    def getJoinString(dof):
        dof = dof - 1
        out = "JOIN("
        for i in range(dof):
            out += "${}, ".format(i)
        out += "${})".format(dof)
        return out

if __name__ == "__main__":
    # Ejemplo
    # Setting up this
    gen = XmlGenerator(3)
    gen.setUpDomainNodes(nodesPerDim=[41,41,41])
    gen.generateXMLTemplate()

    # The data is passed by solver step converged from here
    meshGrid = gen.generateMeshData("mesh1")
    gen.setTimeStamp(0.123456, meshGrid)
    gen.setAttribute("velocity", 99, meshGrid)

    # final print out
    # gen.writeFile(nameFile="testing-1")
    gen.printify()
