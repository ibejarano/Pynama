from petsc4py import PETSc
import logging

class Paraviewer:
    def __init__(self, comm):
        self.comm = comm

    def saveMesh(self, coords, name='mesh'):
        coords.setName(name) #coords must be a PETSc vec obj
        ViewHDF5 = PETSc.Viewer()     # Init. Viewer
        ViewHDF5.createHDF5(name + '.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)

        # self.fullCoordVec.view()
        ViewHDF5.view(obj=coords)  # Put PETSc object into viewer
        ViewHDF5.destroy()

    def saveVec(self, vec, timeStep=None):
        """Save the vector."""
        self.logger.debug("saveVec %s" % name)
        ViewHDF5 = PETSc.ViewerHDF5()     # Init. Viewer

        name = vec.getName()
        if timeStep is None:
            ViewHDF5.create(name + '.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)
        else:
            ViewHDF5.create(name + '-%04d.h5' % timeStep,
                            mode=PETSc.Viewer.Mode.WRITE, comm=self.comm)
        ViewHDF5.pushGroup('/fields')
        ViewHDF5.view(obj=vec)   # Put PETSc object into the viewer
        ViewHDF5.destroy()            # Destroy Viewer

    def saveXmf(self, name, numItems=None, meshName='mesh'):
        """Save the XMF related to the total number of nodes."""
        self.logger.debug("saveXmf")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts).4E" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XY">
        <DataItem Dimensions="%(nodesX2)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>

    <Attribute Name="%(name)s" AttributeType="Scalar" Center="Node">
        <DataItem NumberType="Float" Format="HDF" Dimensions="%(nodes)d">
            %(name)s-%(stepnum)04d.h5:/fields/%(name)s
        </DataItem>
    </Attribute>
</Grid>

"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

    def saveXmfMult(self, name, numItems=None, meshName='mesh', variables=None):
        """Save the XMF related to the total number of nodes."""
        self.logger.debug("saveXmf")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts).4E" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XY">
        <DataItem Dimensions="%(nodesX2)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>
    """
        attributes = """
    <Attribute Name="%(name)s" AttributeType="Scalar" Center="Node">
        <DataItem NumberType="Float" Format="HDF" Dimensions="%(nodes)d">
            %(name)s-%(stepnum)04d.h5:/fields/%(name)s
        </DataItem>
    </Attribute>
"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

        if numItems is None:
            numItems = self.nnodetot

        with open(name+'.xmf', 'w') as fout:
            fout.write(header)
            if not self._ts:
                self._ts = [0]
            for i, ts in enumerate(self._ts):
                fout.write(grid % { 'ts': ts,
                                    'nodes': numItems,
                                    'nodesX2': numItems * 2,
                                    'meshName': meshName
                                    })
                for var in variables:
                    fout.write(attributes % {
                        'nodes': numItems,
                        'stepnum': i,
                        'name': var,
                        })
                fout.write("""</Grid>
""")
            fout.write(footer)

    def saveXmf2(self, name, numItems=None, meshName='mesh'):
        """Save the XMF related to the total number of nodes."""
        self.logger.debug("saveXmf2")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts)s" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XY">
        <DataItem Dimensions="%(nodesX2)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>

    <Attribute Name="%(name)s" AttributeType="Vector" Center="Node">
        <DataItem ItemType="Function" Dimensions="%(nodes)d 3"
                Function="JOIN($0, $1, 0*$1)">

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sX">
                <DataItem Dimensions="3 1" Format="XML">
                    0
                    2
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX2)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sY">
                <DataItem Dimensions="3 1" Format="XML">
                    1
                    2
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX2)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

        </DataItem>
    </Attribute>
</Grid>

"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

        if numItems is None:
            numItems = self.nnodetot

        with open(name+'.xmf', 'w') as fout:
            fout.write(header)
            if((not hasattr(self, '_ts')) or (not self._ts)):
                self._ts = [0]
            for i, ts in enumerate(self._ts):
                fout.write(grid % {'ts': ts, 'nodes': numItems,
                                   'nodesX2': numItems * 2, 'stepnum': i,
                                   'name': name, 'meshName': meshName})
            fout.write(footer)

    def saveXmf2Mult(self, name, numItems=None, meshName='mesh', variables=None):
        """Save the XMF related to the total number of nodes."""
        self.logger.debug("saveXmf2")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts)s" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XY">
        <DataItem Dimensions="%(nodesX2)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>
    """
        attributes = """
    <Attribute Name="%(name)s" AttributeType="Vector" Center="Node">
        <DataItem ItemType="Function" Dimensions="%(nodes)d 3"
                Function="JOIN($0, $1, 0*$1)">

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sX">
                <DataItem Dimensions="3 1" Format="XML">
                    0
                    2
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX2)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sY">
                <DataItem Dimensions="3 1" Format="XML">
                    1
                    2
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX2)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

        </DataItem>
    </Attribute>
"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

        if numItems is None:
            numItems = self.nnodetot

        with open(name+'.xmf', 'w') as fout:
            fout.write(header)
            if not self._ts:
                self._ts = [0]
            for i, ts in enumerate(self._ts):
                fout.write(grid % { 'ts': ts,
                                    'nodes': numItems,
                                    'nodesX2': numItems * 2,
                                    'meshName': meshName
                                    })
                for var in variables:
                    fout.write(attributes % {
                        'nodes': numItems,
                        'nodesX2': numItems * 2,
                        'stepnum': i,
                        'name': var,
                        })
                fout.write("""</Grid>
""")
            fout.write(footer)

    def saveXmf3(self, name, numItems=None, meshName='mesh'):
        """Save the XMF file for attributes with 3 components."""
        self.logger.debug("saveXmf3")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts)s" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XYZ">
        <DataItem Dimensions="%(nodesX3)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>

    <Attribute Name="%(name)s" AttributeType="Vector" Center="Node">
        <DataItem ItemType="Function" Dimensions="%(nodes)d 3"
                Function="JOIN($0, $1, $2)">

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sX">
                <DataItem Dimensions="3 1" Format="XML">
                    0
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sY">
                <DataItem Dimensions="3 1" Format="XML">
                    1
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sZ">
                <DataItem Dimensions="3 1" Format="XML">
                    2
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

        </DataItem>
    </Attribute>
</Grid>

"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

        if numItems is None:
            numItems = self.nnodetot

        with open(name+'.xmf', 'w') as fout:
            fout.write(header)
            if not self._ts:
                self._ts = [0]
            for i, ts in enumerate(self._ts):
                fout.write(grid % {'ts': ts, 'nodes': numItems,
                                   'nodesX2': numItems * 2,
                                   'nodesX3': numItems * 3, 'stepnum': i,
                                   'name': name, 'meshName': meshName})
            fout.write(footer)


    def saveXmf3Mult(self, name, numItems=None, meshName='mesh', variables=None):
        """Save the XMF file for attributes with 3 components."""
        self.logger.debug("saveXmf3")
        header = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""
        grid = """        <Grid Name="mesh1" GridType="Uniform">
    <Time Value="%(ts)s" />
    <Topology TopologyType="Polyvertex" Dimensions="%(nodes)d"/>
    <Geometry GeometryType="XYZ">
        <DataItem Dimensions="%(nodesX3)d" NumberType="Float" Format="HDF">
            %(meshName)s.h5:/fields/%(meshName)s
        </DataItem>
    </Geometry>
"""
        attributes ="""     <Attribute Name="%(name)s" AttributeType="Vector" Center="Node">
        <DataItem ItemType="Function" Dimensions="%(nodes)d 3"
                Function="JOIN($0, $1, $2)">

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sX">
                <DataItem Dimensions="3 1" Format="XML">
                    0
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sY">
                <DataItem Dimensions="3 1" Format="XML">
                    1
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

            <DataItem ItemType="HyperSlab" Dimensions="%(nodes)d"
                    Name="%(name)sZ">
                <DataItem Dimensions="3 1" Format="XML">
                    2
                    3
                    %(nodes)d
                </DataItem>
                <DataItem Dimensions="%(nodesX3)d" NumberType="Float"
                        Format="HDF">
                    %(name)s-%(stepnum)04d.h5:/fields/%(name)s
                </DataItem>
            </DataItem>

        </DataItem>
    </Attribute>
"""
        footer = """    </Grid>
</Domain>
</Xdmf>
"""

        if numItems is None:
            numItems = self.nnodetot

        with open(name+'.xmf', 'w') as fout:
            fout.write(header)
            if not self._ts:
                self._ts = [0]
            for i, ts in enumerate(self._ts):
                fout.write(grid % { 'ts': ts,
                                    'nodes': numItems,
                                    'nodesX3': numItems * 3,
                                    'meshName': meshName
                                    })
                for var in variables:
                    fout.write(attributes % {
                        'nodes': numItems,
                        'nodesX3': numItems * 3,
                        'stepnum': i,
                        'name': var,
                        })
                fout.write("""</Grid>
""")
            fout.write(footer)