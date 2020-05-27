from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

class XmlGenerator(object):
    def __init__(self):
        pass

    def getXMLTemplate(self):
        root = Element('Xdmf')
        root.set('Version', '2.0')
        # comment = Comment('Generated for PyMOTW')
        # top.append(comment)
        gridTime = Element('Grid')
        gridTime.set('Name', 'mesh1')
        gridTime.set('GridType', "uniform")
        domain = SubElement(root, 'Domain')
        grid = SubElement(domain, 'Grid')
        grid.set('Name',"TimeSeries")
        grid.set('GridType',"Collection")
        grid.set('CollectionType',"Temporal")
        
        return root

    def printMesh(self,):
        pass

    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

a = XmlGenerator()
root = a.getXMLTemplate()
print(a.prettify(root))

