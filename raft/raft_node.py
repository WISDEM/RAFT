# RAFT's node class

import numpy as np
import matplotlib.pyplot as plt
from raft.helpers import getH

class Node:
    ''' This class is used to represent nodes in the FOWT, which are responsible for describing the motions of the structure.'''

    def __init__(self, node_id, r0, nw, member=None, end_node=True):
        ''' 
        Initialize a Node object.

        PARAMETERS
        ----------
        node_id : int
            Unique identifier for the node.
        r0 : float array
            If 3-component array, this is the position of the node wrt to the platform reference point (PRP).
            If 6-component array, this is the position of the node wrt to the PRP, and the initial attitude (roll, pitch, yaw).
        member : Member object, optional
            Reference to the member that this node belongs to. Not all nodes are part of a member, e.g. nodes that are part of a rigid link will have this set to None.
        end_node : bool, optional
            True if this is an end node of a member, False if this is an internal node of a flexible member. Default is True.
        '''
        self.id = node_id
        self.nTransDOF = 3 # Number of translation dofs per node
        self.nRotDOF   = 3 # Number of rotational dofs
        self.nDOF      = self.nTransDOF + self.nRotDOF # Total number of dofs per node
        self.member = member # Reference to the member that this node belongs to. If part of a rigid link, this will be None once we start using RAFT members

        # Node position + attitude of the node relative to PRP [m, rad]
        if len(r0) == 3:
            self.r0  = np.hstack((r0, np.zeros(self.nRotDOF)))
        elif len(r0) == 6:
            self.r0  = r0.copy()
        else:
            raise Exception(f"Node {node_id} position must be a 3- or 6-component vector, but got {len(r0)} components instead.")

        self.r  = self.r0.copy() # Current position and attitude of the node in the global reference frame (current wrp to PRP but updated when updating the FOWT's position)
        self.Xi0 = np.zeros(self.nDOF) # mean offsets of the node from its reference point [m, rad]        
        self.end_node = end_node              # True if this is an end node of a member, False if this is an internal node of a flexible member
        
        # Things with `None` will be assigned later
        self.nodeList      = None # Reference to the list of nodes in the structure. Assigned when initializing the Structure object.
        self.T             = None # Transformation matrix that relates the 6 dofs of this node to the reduced dofs of the STRUCTURE (rows of structure.T that correspond to this node)
        self.parentNode_id = None # ID of the parent node. Assigned when attaching this node to another node.
        self.reducedDOF    = None # Reduced DoFs that are needed to describe this node. This is a subset of the reduced dofs of the whole structure. List of lists with two elements: [node_id, dof_id]. E.g., [[3, 5]] is the rotation around z of node with id equal to 3.
        self.T_aux         = None # Subset of the transformation matrix T. T_aux relates the 6 dofs of this node to a subset of the reduced dofs required to describe this node only.          
        # self.Xi            = None # complex response amplitudes as a function of frequency  [m, rad]        

        self.joint_id      = None
        self.joint_type    = None
        self.rigid_link_id = None

    def getRigidConnectedNode(self):
        '''
        If this node is part of a rigid link, returns the other node that is part of the rigid link.
        '''
        # Does not apply if node is not part of a rigid link
        if self.rigid_link_id is None:
            return None

        # Checking if two nodes: 1. Are part of the same rigid link, and 2. Have different ids          
        rigidConnectedNode = [n for n in self.nodeList if (n.rigid_link_id == self.rigid_link_id and n.id != self.id)]
        if len(rigidConnectedNode) != 1:
            raise Exception(f"Node {self.id} rigidly connected to {len(rigidConnectedNode)} nodes, but rigidly links can contain only two nodes.")
        return rigidConnectedNode[0]

    def getNodesConnectedByJoint(self):
        '''
            Returns a list of nodes that are connected to this Node object by a joint.
            This Node object needs to have a reference to the list of nodes in the structure (self.nodeList).
        '''
        if self.joint_id is None:
            return []
        return [n for n in self.nodeList if (n.joint_id == self.joint_id and n.id != self.id)]

    def attachToNode(self, node, rigid_link=False):
        '''
        Attach this node to another node. This is done by assigning
        > `self.reducedDOF`: the set of reduced dofs needed to describe the motions of this node. A subset of the reduced dofs of the whole structure
        > `self.T_aux`: transformation matrix, with nDOF rows and len(self.reducedDOF) columns, that relates the reduced dofs of this node to the full dofs. A subset of the transformation matrix T of the whole structure
        > `self.parentNode_id`: ID of the node to which this node is attached
        based on the reduced dofs of the other node and the joint type.
        
        PARAMETERS
        ----------
        node : Node object
            The node to which this node will be attached, i.e., the node whose dofs will be copied to self based on the joint type.
        rigid_link : bool, optional
            If True, this node will be attached to the other node using a rigid link.
            If False, it will be attached using self.joint_type that was previously assigned to this node.
        '''
        # Only end nodes can be attached to other nodes using joints or rigid links
        if not self.end_node:
            return
        
        # Check if self is not the parent of the node to which we are trying to attach
        if node.parentNode_id == self.id:
            return
        
        dofs2assign = node.reducedDOF.copy() # `node.reducedDOF` is a subset of the reducedDOF of the structure (just the dofs that describe the motions of `node`)
        T2assign = node.T_aux.copy() # `T_aux` is a subset of `self.T`, corresponding to the reducedDoFs needed to describe this node only        

        if rigid_link:
            joint_type = "rigid_link"
        else:
            joint_type = self.joint_type
            if joint_type is None:
                raise Exception(f"Node {self.id} has no joint assigned. Cannot attach to another node.")

        if joint_type == "rigid_link":
            # The rigid link imposes that the nodes move as a rigid body. This corresponds to a linear transformation where
            # > both nodes have the same rotation, 
            # > the translation of one node is equal to the translation of the other node + the cross product of the rotation and the distance vector between the two nodes
            rotation = node.T_aux[self.nTransDOF:self.nDOF, :] # 'rotation' that corresponds to the rotations written in the reduced dofs
            T2assign[:self.nTransDOF, :] += getH(self.r - node.r) @ rotation

        # If ball joint, this node starts with the same reduced dofs as the input node + its own rotation
        # We remove some dofs if their columns are zero in the transformation matrix
        elif joint_type == "ball" or joint_type == "universal":            
            T2assign = np.hstack((T2assign, np.zeros((T2assign.shape[0], self.nRotDOF)))) # Add columns that correspond to the rotation of this node
            T2assign[self.nTransDOF:self.nDOF,:] = 0 # Make the rotation of this node independent of previous rotations 
            for idof in range(self.nTransDOF, self.nDOF):
                dofs2assign += [[self.id, idof]] # Rotation around dof i
                T2assign[idof, dofs2assign.index([self.id, idof])] = 1
            

            # Remove columns with zeros. Need to remove the corresponding dof from the list as well
            for i in range(T2assign.shape[1]-1, -1, -1): # Iterate starting from the last column and going to the first column index (inclusive)
                if np.all(T2assign[:, i] == 0):
                    T2assign = np.delete(T2assign, i, axis=1)
                    dofs2assign.pop(i)
        
        # If cantilever, self has the same dofs as the input node and the same transformation matrix,
        # so we were done already when we did `dofs2assign = node.reducedDOF.copy()` and `T2assign = node.T_aux.copy()`
        # Leaving this `pass` here to make it clear that we are not doing anything else.
        elif joint_type == 'cantilever':
            pass
        
        # Sort dofs2assign based on node ID first and then by dof
        sorted_indices = sorted(range(len(dofs2assign)), key=lambda i: (dofs2assign[i][0], dofs2assign[i][1]))
        dofs2assign = [dofs2assign[i] for i in sorted_indices]
        T2assign = T2assign[:, sorted_indices]

        # For rigidly linked nodes, the rotation of one node is equal to the rotation of the other node.
        # We always keep the rotation of the node with the smallest ID to avoid the same dof appearing twice under different "names".
        for idof, dof in enumerate(dofs2assign):
            eqDof = self.findEquivalentDof(dof)
            if (eqDof is not None) and (eqDof[0] <= dof[0]):
                dofs2assign[idof] = eqDof

        # If `self` did not have reduced dofs assigned yet, we can simply assign the dofs determined above  
        if self.reducedDOF is None:
            self.reducedDOF = dofs2assign.copy()
            self.T_aux = T2assign.copy()
            self.parentNode_id = node.id
            return
        
        # THE CODE BELOW IS FOR CLOSED KINEMATIC LOOPS.
        # I do not trust this part of the code. I did some tests with a simple 2D code and it seems to work, but do not rely on it before it is properly tested
        # For now, it's best to avoid closed kinematic loops, for example using flexible elements.            
        #
        # If the node already had reduced dofs assigned (when we have a closed kinematic loop), we need to ensure compatibility with the dofs that we are trying to assign.
        if dofs2assign != self.reducedDOF:
            # The incompatibility is between two sets of reduced dofs. We need to know which set has more dofs than the other.
            # It doesn't matter which set is longer (if the one already assined or the one that we're trying to assing), 
            # we just need to distinguish between the long list and the short list of dofs
            if len(dofs2assign) >= len(self.reducedDOF):
                longListDofs, shortListDofs = dofs2assign.copy(), self.reducedDOF.copy()
                T_long, T_short = T2assign.copy(), self.T_aux.copy()
            else:
                longListDofs, shortListDofs = self.reducedDOF.copy(), dofs2assign.copy()
                T_long, T_short = self.T_aux.copy(), T2assign.copy()

            # In the end, we want to keep the short list of dofs. 
            # The other dofs are redundant and will be removed below.
            self.reducedDOF = shortListDofs.copy()
            self.T_aux = T_short.copy()

            # The long list of dofs (or redundant list) can be written as a linear combination of the short list of dofs,
            # i.e. T_long = compatMatrix @ T_short
            # We want to compute the compatibility matrix compatMatrix
            compatMatrix = np.zeros((T_long.shape[0], T_short.shape[1])) 
            commonDOF = [d for d in shortListDofs if d in longListDofs] # Get the indices of the dofs that are in both lists. These will be assigned 1 in the compatibility matrix

            # Get the part of the short matrix that is not repeated in the long matrix - the rest is filled with zeros
            T_short_noCommon = np.zeros(T_short.shape)
            for i, dof in enumerate(shortListDofs):
                if (dof not in commonDOF):
                    T_short_noCommon[:, i] = T_short[:, i].copy()
                else: # The dof can be in both but with a different column in the T matrix (i.e. different contribution to the dofs of the node)
                    i_long = longListDofs.index(dof)       
                    if not np.all(T_long[:, i_long] == T_short[:, i]):
                        T_short_noCommon[:, i] = T_short[:, i].copy()
                    
            # Same for the long matrix
            T_long_noCommon = np.zeros(T_long.shape)
            for i, dof in enumerate(longListDofs):
                if (dof not in commonDOF):
                    T_long_noCommon[:, i] = T_long[:, i].copy()
                else: # The dof can be in both but with a different column in the T matrix (i.e. different contribution to the dofs of the node)
                    i_short = shortListDofs.index(dof)                    
                    if not np.all(T_short[:, i_short] == T_long[:, i]):
                        T_long_noCommon[:, i] = T_long[:, i].copy()

            # Solve the linear system for the parts of the compatibility matrix that are not directly related to each other
            try:
                T_long_noCommon_inv = np.linalg.inv(T_long_noCommon)
            except np.linalg.LinAlgError:
                T_long_noCommon_inv = np.linalg.pinv(T_long_noCommon)
            compatMatrix = T_long_noCommon_inv@T_short_noCommon

            # Fill with 1 the parts that are directly related to each other but only if they were not filled before
            for irow, dof in enumerate(longListDofs):                
                if dof in commonDOF:
                    icol = shortListDofs.index(dof)
                    compatMatrix[irow, icol] = 1 if compatMatrix[irow, icol] == 0 else compatMatrix[irow, icol]


            # Loop all nodes to ensure compatibility
            # Basically, we loop the dofs of longListDofs and replace them in the reduced list of dofs
            # by the linear combination of the dofs of shortListDofs
            for n in self.nodeList:
                if n.reducedDOF is not None: # Only if reduced dofs were already assigned
                    for irowIn, dofLong in enumerate(longListDofs): # Loop through the dofs of the long list
                        if dofLong in n.reducedDOF: # Only need to do something if this dof is part of the reduced dofs of the node                            
                            icolOut = n.reducedDOF.index(dofLong) # Index of the dof in the reduced dofs of the node n
                            col2multiply = n.T_aux[:, icolOut].copy() # Column of the transformation matrix that corresponds to this dof
                            n.T_aux[:, icolOut] = 0 # Clear the column to fill it with the linear combination of the dofs of the short list                            

                            # Loop through the dofs of the short list
                            for icolIn, dofShort in enumerate(shortListDofs):
                                if dofShort not in n.reducedDOF:
                                    n.reducedDOF.append(dofShort)
                                    n.T_aux = np.hstack((n.T_aux, np.zeros((n.T_aux.shape[0], 1))))
                                iColOut = n.reducedDOF.index(dofShort) # Index of the dof in the reduced dofs of the node n
                                for irowOut in range(n.T_aux.shape[0]):
                                    # n.T_aux[:, iColOut] = T_short[:, icolIn] + compatMatrix[irowIn, icolIn] * T_long[:, icolOut]
                                    n.T_aux[irowOut, iColOut] += compatMatrix[irowIn, icolIn] * col2multiply[irowOut]

                    # Remove columns with zeros. Need to remove the corresponding dof from the list as well
                    for i in range(n.T_aux.shape[1]-1, -1, -1): # Iterate starting from the last column and going to the first column index (inclusive)
                        if np.all(n.T_aux[:, i] == 0):
                            n.T_aux = np.delete(n.T_aux, i, axis=1)
                            n.reducedDOF.pop(i)

    def findEquivalentDof(self, dof):
        # TODO: I don't think we need this anymore. After we have a nice set of tests, remove this function to see if things change.
        # Equivalent dofs happen for rigid links, where the rotation of one node is equal to the rotation of the other node        
        eqDof = None

        if dof[1] == 3 or dof[1] == 4 or dof[1] == 5: # Can only happen for rotation
            nID = dof[0] # ID of the node corresponding to the input dof
            n   = [n for n in self.nodeList if n.id == nID][0] # Find the node itself            
            rigidConnectedNode = n.getRigidConnectedNode()            
            if rigidConnectedNode is not None:
                eqDof = [rigidConnectedNode.id, dof[1]]
        return eqDof
        
    def setT(self, reducedDoFs_structure):
        '''
        Set the transformation matrix T that relates the nDOF dofs of this node to the 
        reduced set of dofs of the whole FOWT to which this node belongs, self.T
                
        PARAMETERS
        ----------
        reducedDoFs_structure : list of lists
            Reduced dofs of the whole structure. It has the same format as self.reducedDOF, a list of lists where each sublist has two elements: node ID and dof ID.
        '''
        # Based on the dofs of this node and the matrix T_aux, we can fill the transformation matrix T
        # that relates the full dofs of this node (rows of matrix T) to the reduced dofs of the structure (columns of matrix T).
        # Difference between T and T_aux:
        # - T has a number of columns that corresponds to the list of reduced dofs of the STRUCTURE
        # - T_aux has a number of columns that corresponds to the list of reduced dofs of this node
        if self.T is None:
            self.T = np.zeros((self.T_aux.shape[0], len(reducedDoFs_structure)))
        for i in range(self.T_aux.shape[0]):  # Loop through rows
            # The row in T is the same row in T_aux, as both correspond to the full dofs
            # of the node (0: x, 1: y, 2: z, 3: rotation around x, 4: rotation around y, 5: rotation around z)
            for j in range(self.T_aux.shape[1]):  # Loop through columns
                # The index j corresponds to the list of reduced dofs stored in self.reducedDOF. 
                # self.reducedDOF is a subset of the reduced dofs of the whole structure, so we
                # need to know the index of this dof in the list of reduced dofs of the structure
                try:
                    col = reducedDoFs_structure.index(self.reducedDOF[j])
                    self.T[i, col] = self.T_aux[i, j]
                except ValueError:
                    raise Exception(f"Node {self.id} has a dof that is not part of the reduced dofs of the structure. Why???")

    def setDisplacementLinear(self, reducedDisp_structure):
        '''
        Compute node displacements using the linear transformation, self.T
        
        PARAMETERS
        ----------
        reducedDisp_structure: vector with nReducedDisp components
            Reduced displacements of the whole structure written in the reduced set of dofs of the structure.
        '''
        if self.T is None:
            raise Exception("Transformation matrix T is not set yet. Call setT() before calling setDisplacementLinear().")
        self.Xi0 = self.T @ reducedDisp_structure

    def setPositionLinear(self, reducedDisp_structure):
        # TODO: Remove this function. Set the position within the fowt class
        '''
        Set node position and attitude using the linear transformation, self.T
        
        PARAMETERS
        ----------
        reducedDisp_structure: vector with nReducedDisp components
            Reduced displacements of the whole structure written in the reduced set of dofs of the structure.
        '''
        self.setDisplacementLinear(reducedDisp_structure)
        self.r = self.r0 + self.Xi0

    def plot(self, ax=None, color='default', size=5, marker='o', markerfacecolor='default', writeID=False):
        if color == 'default':
            if self.end_node:
                color = 'k'
            else:
                color = 'b'

        if markerfacecolor == 'default':
            markerfacecolor = color

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.r[0], self.r[1], self.r[2], color=color, marker=marker, s=size, facecolors=markerfacecolor)
        
        # Convert marker size from points to data units
        fig = ax.get_figure()
        d = (size / 72.0) * fig.dpi / ax.transData.transform((1, 0))[0]

        if writeID:
            ax.text(self.r[0], self.r[1], self.r[2], str(self.id), color=color)
        return ax