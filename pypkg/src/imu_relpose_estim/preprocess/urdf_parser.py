import xml.etree.ElementTree as ET
import numpy as np
from imu_relpose_estim.utils.rotation_helper import RotationHelper

class LinkTreeNode:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

class UrdfLinkTree:
    def __init__(self, 
                 node_list: LinkTreeNode,
                 links):
        self.node_list = node_list
        self.nodelist_with_depth_index = self._get_nodelist_with_depth_index()
        self.all_links = links

    def find_all_nodenames(self):
        name_list = []
        for n in self.node_list:
            if not (n.name in name_list):
                name_list.append(n.name)
            if not (n.parent in name_list):
                name_list.append(n.parent)
        return name_list
    
    def search_parent(self, name):
        for n in self.node_list:
            if n.name == name:
                return n.parent
        return None
    
    def search_children(self, name):
        children = []
        for n in self.node_list:
            if n.parent == name:
                children.append(n.name)
        return children
    
    def trace_to_root(self, name):
        trace = [name]
        current_name = name
        while True:
            parent = self.search_parent(current_name)
            if parent is None:
                break
            trace.append(parent)
            current_name = parent
        return trace
    

    def find_root(self):
        init = self.node_list[0].parent
        trace = self.trace_to_root(init)
        return trace[-1]
    

    def _get_nodelist_with_depth_index(self):
        unvisited_nodes = self.find_all_nodenames()
        number_of_nodes = len(unvisited_nodes)
        root = self.find_root()

        ## initialize
        unvisited_nodes.remove(root)
        current_name = root
        depth = 0
        visited_nodes = [root]
        depth_list = [0]

        for i in range(number_of_nodes - 1):
            children = self.search_children(current_name)
            if not children == []:
                ## if node is not leaf
                depth += 1
                for child in children:
                    if child in unvisited_nodes:
                        visited_nodes.append(child)
                        depth_list.append(depth)
                        ## remove from unvisited node queue
                        unvisited_nodes.remove(child)
            
            ## visit to previous child nodes
            current_name = visited_nodes[i+1]

        return list(zip(visited_nodes, depth_list))


    def sort_by_depth(self, name_list):
        depths = []
        names = []
        for n, d in self.nodelist_with_depth_index:
            if n in name_list:
                names.append(n)
                depths.append(d)

        max_depth = max(depths)
        
        sorted_depths = []
        sorted_name = []
        for i in range(max_depth + 1):
            if i in depths:
                sorted_name.append(names[depths.index(i)])
                sorted_depths.append(i)

        return sorted_name, sorted_depths
    
    @classmethod
    def make_tree_from_robot_description(cls, robot_description):
        root = ET.fromstring(robot_description)
        links = root.findall("link")
        joints = root.findall("joint")

        ## parse joints
        nodes = []
        for j in joints:
            nodes.append(LinkTreeNode(j.find("child").attrib["link"], j.find("parent").attrib["link"]))
        
        ## create tree object
        tree = cls(nodes, links)
        return tree
    
    
    @classmethod
    def parse_imu_names_from_robot_description(cls, robot_description):
        tree = cls.make_tree_from_robot_description(robot_description)
        all_imu_names = [l.attrib["name"] for l in tree.all_links if "imu" in l.attrib["name"]]

        ## sorted by distance from root
        sorted_imu_names, _ = tree.sort_by_depth(all_imu_names)

        return sorted_imu_names
    

    @classmethod
    def get_nodelist_with_depth_index(cls, robot_description):
        tree = cls.make_tree_from_robot_description(robot_description)
        return tree.nodelist_with_depth_index
    

    @staticmethod
    def find_parent_ordinallink(imu_linkname, all_link_names_with_depth):
        ## helper
        def linkname_helper(imu_name):
            if "__link__" in imu_name:
                return False
            if "_parent_imu" in imu_name:
                return False
            if "_child_imu" in imu_name:
                return False
            return True

        depth_target = -1
        for name, depth in all_link_names_with_depth[::-1]:
            if imu_linkname == name:
                depth_target = depth
            
            if depth < depth_target and linkname_helper(name):
                return name


    @staticmethod
    def get_all_IMU_link_pose_wrt_assoc_joint(symbolic_urdf):
        ## helper
        def origin_elem_to_homogeneous_matrix(origin_elem: ET.Element):
            ## get xyz and rpy from parsed xml
            xyz_str = origin_elem.attrib["xyz"]
            rpy_str = origin_elem.attrib["rpy"]

            ## convert them into ndarray
            ## NOTE: omitting empty string is for allowing indents or multiple spaces in symbolic urdf file
            xyz_array = np.array([float(s) for s in xyz_str.split(" ") if not s == ""])
            rpy_array = np.array([float(s) for s in rpy_str.split(" ") if not s == ""])

            ## get homogeneous matrix
            Hmat = np.eye(4)
            Hmat[:3,:3] = RotationHelper.rpy_to_rotmat(rpy_array)
            Hmat[:3,3]  = xyz_array

            return Hmat

        # print(symbolic_urdf)
        root = ET.fromstring(symbolic_urdf)
        joints = root.findall("joint")

        all_imu_fixedjoints = [j for j in joints if "imu_fixedjoint" in j.attrib["name"]]

        return dict([(j.find("child").attrib["link"], origin_elem_to_homogeneous_matrix(j.find("origin"))) for j in all_imu_fixedjoints])

