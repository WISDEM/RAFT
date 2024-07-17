# This is a file to handle meshing of things in RAFT for the time being. It outputs in the HAMS .pnl format.

import numpy as np
import os
import os.path as osp

def makepanel(X, Y, Z, savedNodes, savedPanels):
    '''
    Sets up panel and node data for HAMS .pnl input file.
    Also ensures things don't go above the water line. (A rough implementation that should be improved.)

    X, Y, Z : lists
        panel coordinates - 4 expected
    savedNodes : list of lists
        all the node coordinates already saved
    savedPanels : list
        the information for all the panels already saved: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID  Vertex3_ID   (Vertex4_ID)

    '''
    
    
    # if the panel is fully out of the water, skip it
    if (np.array(Z) > 0.0).all():   
        return
    
    # if any points are above the water, bring them down to the water surface
    for i in range(4):
        if Z[i] > 0.0:
            Z[i] = 0.0
            
    # now process the node points w.r.t. existing list
    #points = np.vstack([X, Y, Z])                # make a single 2D array for easy manipulation
    
    pNodeIDs = []                                    # the indices of the nodes for this panel (starts at 1)
    
    pNumNodes = 4
    
    for i in range(4):
    
        ndi = [X[i],Y[i],Z[i]]                       # the current node in question in the panel
        
        match = [j+1 for j,nd in enumerate(savedNodes) if nd==ndi]  # could do this in reverse order for speed...
        
        if len(match)==0:                            # point does not already exist in list; add it.
            savedNodes.append(ndi)
            pNodeIDs.append(len(savedNodes))
            
        elif len(match)==1:                          # point exists in list; refer to it rather than adding another point
            
            if match[0] in pNodeIDs:                 # if the current panel has already referenced this node index, convert to tri
                #print("triangular panel detected!")  # we will skip adding this point to the panel's indix list                                                     
                pNumNodes -= 1                       # reduce the number of nodes for this panel
                
            else:
                pNodeIDs.append(match[0])            # otherwise add this point index to the panel's list like usual
            
        else:
            ValueError("Somehow there are duplicate points in the list!")
            
    panelID = len(savedPanels)+1                     # id number of the current panel (starts from 1)
    
    
    if pNumNodes == 4:
        savedPanels.append([panelID, pNumNodes, pNodeIDs[0], pNodeIDs[1], pNodeIDs[2], pNodeIDs[3]])
    elif pNumNodes == 3:
        savedPanels.append([panelID, pNumNodes, pNodeIDs[0], pNodeIDs[1], pNodeIDs[2]])
    else:
        ValueError(f"Somehow there are only {pNumNodes} unique nodes for panel {panelID}")

def meshRectangularMember(stations, widths, heights, rA, rB, dz_max=0, da_max=0, dh_max=0, savedNodes=[], savedPanels=[], endA=True, endB=True):
    
    half_widths = 0.5 * np.array(widths)
    half_heights = 0.5 * np.array(heights)

    # Step through each station and create vertices
    for i_s in range(len(stations)):
        w = half_widths[i_s]
        h = half_heights[i_s]
        z = stations[i_s]

        # Define the four vertices of the rectangular cross-section
        savedNodes.append([-w, -h, z])
        savedNodes.append([w, -h, z])
        savedNodes.append([w, h, z])
        savedNodes.append([-w, h, z])

    # Create panels by connecting vertices
    num_vertices_per_station = 4
    for i in range(len(stations) - 1):
        idx0 = i * num_vertices_per_station
        idx1 = (i + 1) * num_vertices_per_station

        savedPanels.append([idx0, idx0 + 1, idx1 + 1, idx1])  # Bottom panel
        savedPanels.append([idx0 + 1, idx0 + 2, idx1 + 2, idx1 + 1])  # Right panel
        savedPanels.append([idx0 + 2, idx0 + 3, idx1 + 3, idx1 + 2])  # Top panel
        savedPanels.append([idx0 + 3, idx0, idx1, idx1 + 3])  # Left panel

    # Mesh end A if requested
    if endA:
        idx0 = 0
        savedPanels.append([idx0, idx0 + 1, idx0 + 2, idx0 + 3])  # End A panel
        savedPanels[-1] = savedPanels[-1][::-1]

    # Mesh end B if requested
    if endB:
        idx0 = (len(stations) - 1) * num_vertices_per_station
        savedPanels.append([idx0, idx0 + 1, idx0 + 2, idx0 + 3])  # End B panel

    # Subdivide panels
    subdivided_vertices = []
    subdivided_panels = []

    for panel in savedPanels:
        v0, v1, v2, v3 = [savedNodes[idx] for idx in panel]

        for i in range(dh_max):
            for j in range(da_max):
                t0 = i / dh_max
                t1 = (i + 1) / dh_max
                s0 = j / da_max
                s1 = (j + 1) / da_max

                p0 = (1 - t0) * (1 - s0) * np.array(v0) + t0 * (1 - s0) * np.array(v1) + t0 * s0 * np.array(v2) + (1 - t0) * s0 * np.array(v3)
                p1 = (1 - t1) * (1 - s0) * np.array(v0) + t1 * (1 - s0) * np.array(v1) + t1 * s0 * np.array(v2) + (1 - t1) * s0 * np.array(v3)
                p2 = (1 - t1) * (1 - s1) * np.array(v0) + t1 * (1 - s1) * np.array(v1) + t1 * s1 * np.array(v2) + (1 - t1) * s1 * np.array(v3)
                p3 = (1 - t0) * (1 - s1) * np.array(v0) + t0 * (1 - s1) * np.array(v1) + t0 * s1 * np.array(v2) + (1 - t0) * s1 * np.array(v3)

                idx_base = len(subdivided_vertices)
                subdivided_vertices.extend([p0, p1, p2, p3])
                subdivided_panels.append([idx_base, idx_base + 1, idx_base + 2, idx_base + 3])

    subdivided_vertices = np.array(subdivided_vertices)

    # Transform coordinates to reflect specified rA and rB values
    rAB = np.array(rB) - np.array(rA)  # displacement vector from end A to end B [m]
    beta = np.arctan2(rAB[1], rAB[0])  # member incline heading from x axis
    phi = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])  # member incline angle from vertical

    # Trig terms for Euler angles rotation based on beta, phi, and gamma
    s1 = np.sin(beta)
    c1 = np.cos(beta)
    s2 = np.sin(phi)
    c2 = np.cos(phi)
    s3 = np.sin(0.0)
    c3 = np.cos(0.0)

    R = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                  [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                  [-c3 * s2, s2 * s3, c2]])  # Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    transformed_nodes = np.matmul(R, subdivided_vertices.T).T + np.array(rA)

    # Add transformed nodes to savedNodes if not already present
    for node in transformed_nodes:
        node_list = list(node)
        if node_list not in savedNodes:
            savedNodes.append(node_list)

    # Adjust each panel position based on member pose then set up the member
    nSavedPanelsOld = len(savedPanels)
    for panel in subdivided_panels:
        v0, v1, v2, v3 = [transformed_nodes[idx] for idx in panel]
        savedPanels.append([savedNodes.index(list(v0)), savedNodes.index(list(v1)), savedNodes.index(list(v2)), savedNodes.index(list(v3))])

    print(f'Of {len(subdivided_panels)} generated panels, {len(savedPanels) - nSavedPanelsOld} were submerged and have been used in the mesh.')

    return savedNodes, savedPanels


def writeMesh(savedNodes, savedPanels, oDir):
    '''Creates a HAMS .pnl file based on savedNodes and savedPanels lists'''
        
    numPanels   = len(savedPanels)
    numNodes    = len(savedNodes)    
        
    # write .pnl file
    if osp.isdir(oDir) is not True:
        os.makedirs(oDir)
        
    oFilePath = os.path.join(oDir, 'HullMesh.pnl')

    oFile = open(oFilePath, 'w')
    oFile.write('    --------------Hull Mesh File---------------\n\n')
    oFile.write('    # Number of Panels, Nodes, X-Symmetry and Y-Symmetry\n')
    oFile.write(f'         {numPanels}         {numNodes}         0         0\n\n')

    oFile.write('    #Start Definition of Node Coordinates     ! node_number   x   y   z\n')
    for i in range(numNodes):
        oFile.write(f'{i+1:>5}{savedNodes[i][0]:18.3f}{savedNodes[i][1]:18.3f}{savedNodes[i][2]:18.3f}\n')
    oFile.write('   #End Definition of Node Coordinates\n\n')
    
    oFile.write('   #Start Definition of Node Relations   ! panel_number  number_of_vertices   Vertex1_ID   Vertex2_ID   Vertex3_ID   (Vertex4_ID)\n')
    for i in range(numPanels):
        oFile.write(''.join([f'{p:>8}' for p in savedPanels[i]])+'\n')
    oFile.write('   #End Definition of Node Relations\n\n')
    oFile.write('    --------------End Hull Mesh File---------------\n')
    
    oFile.close()

def meshRectangularMemberForGDF(stations, widths, heights, rA, rB, dz_max=0, da_max=0, dh_max=0, endA=True, endB=True):

    
    half_widths = 0.5 * np.array(widths)
    half_heights = 0.5 * np.array(heights)

    savedNodes = []
    savedPanels = []

    # Step through each station and create vertices
    for i_s in range(len(stations)):
        w = half_widths[i_s]
        h = half_heights[i_s]
        z = stations[i_s]

        # Define the four vertices of the rectangular cross-section
        savedNodes.append([-w, -h, z])
        savedNodes.append([w, -h, z])
        savedNodes.append([w, h, z])
        savedNodes.append([-w, h, z])

    # Create panels by connecting vertices
    num_vertices_per_station = 4
    for i in range(len(stations) - 1):
        idx0 = i * num_vertices_per_station
        idx1 = (i + 1) * num_vertices_per_station

        savedPanels.append([idx0, idx0 + 1, idx1 + 1, idx1])  # Bottom panel
        savedPanels.append([idx0 + 1, idx0 + 2, idx1 + 2, idx1 + 1])  # Right panel
        savedPanels.append([idx0 + 2, idx0 + 3, idx1 + 3, idx1 + 2])  # Top panel
        savedPanels.append([idx0 + 3, idx0, idx1, idx1 + 3])  # Left panel

    # Mesh end A if requested
    if endA:
        idx0 = stations[0] * num_vertices_per_station
        idx1 = idx0
        savedPanels.append([idx0, idx0 + 1, idx0 + 2, idx0 + 3])  # End A panel
        savedPanels[-1] = savedPanels[-1][::-1]

    # Mesh end B if requested
    if endB:
        idx0 = (len(stations) - 1) * num_vertices_per_station
        idx1 = idx0 + 4
        savedPanels.append([idx0, idx0 + 1, idx0 + 2, idx0 + 3])  # End B panel

    # Subdivide panels
    vertices = []
    subdivided_panels = []

    for panel in savedPanels:
        v0, v1, v2, v3 = [savedNodes[idx] for idx in panel]
        vertices.extend([v0, v1, v2, v3])

        #n_subdivisions = 5  # Example number of subdivisions

        for i in range(dh_max):
            for j in range(da_max):
                t0 = i / dh_max
                t1 = (i + 1) / dh_max
                s0 = j / da_max
                s1 = (j + 1) / da_max

                p0 = (1 - t0) * (1 - s0) * np.array(v0) + t0 * (1 - s0) * np.array(v1) + t0 * s0 * np.array(v2) + (1 - t0) * s0 * np.array(v3)
                p1 = (1 - t1) * (1 - s0) * np.array(v0) + t1 * (1 - s0) * np.array(v1) + t1 * s0 * np.array(v2) + (1 - t1) * s0 * np.array(v3)
                p2 = (1 - t1) * (1 - s1) * np.array(v0) + t1 * (1 - s1) * np.array(v1) + t1 * s1 * np.array(v2) + (1 - t1) * s1 * np.array(v3)
                p3 = (1 - t0) * (1 - s1) * np.array(v0) + t0 * (1 - s1) * np.array(v1) + t0 * s1 * np.array(v2) + (1 - t0) * s1 * np.array(v3)

                idx_base = len(vertices)
                vertices.extend([p0, p1, p2, p3])
                subdivided_panels.append([idx_base, idx_base + 1, idx_base + 2, idx_base + 3])

    vertices = np.array(vertices)

    # Transform coordinates to reflect specified rA and rB values
    rAB = rB - rA  # displacement vector from end A to end B [m]
    beta = np.arctan2(rAB[1], rAB[0])  # member incline heading from x axis
    phi = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])  # member incline angle from vertical

    # Trig terms for Euler angles rotation based on beta, phi, and gamma
    s1 = np.sin(beta)
    c1 = np.cos(beta)
    s2 = np.sin(phi)
    c2 = np.cos(phi)
    s3 = np.sin(0.0)
    c3 = np.cos(0.0)

    R = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                  [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                  [-c3 * s2, s2 * s3, c2]])  # Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    vertices = np.dot(vertices, R.T) + rA

    return vertices, subdivided_panels

def writeMeshToGDF(vertices, panels, filename="platform.gdf", aboveWater=True):

    npan = len(panels)

    with open(filename, "w") as f:
        f.write('gdf mesh \n')
        f.write('1.0   9.8 \n')
        f.write('0, 0 \n')
        f.write(f'{npan}\n')

        if aboveWater:
            for panel in panels:
                for idx in panel:
                    f.write(f'{vertices[idx, 0]:>10.3f} {vertices[idx, 1]:>10.3f} {vertices[idx, 2]:>10.3f}\n')
        else:  # this option avoids making panels above the waterline
            for panel in panels:
                panel_vertices = vertices[panel]
                if any(panel_vertices[:, 2] < -0.001):  # only consider the panel if it's at least partly submerged (some z < 0)
                    for vertex in panel_vertices:
                        if vertex[2] > 0:
                            vertex[2] = 0
                        f.write(f'{vertex[0]:>10.3f} {vertex[1]:>10.3f} {vertex[2]:>10.3f}\n')

        f.close()

if __name__ == "__main__":

    stations = [0, 15, 30, 45]
    widths = [15, 15, 10, 10]
    heights = [15, 15, 10, 10]

    rA = np.array([0, 0, -15])
    rB = np.array([0, 0, 20])
    dz_max = 1
    da_max = 20
    dh_max= 4


    vertices, panels = meshRectangularMemberForGDF(stations, widths, heights, rA, rB, dz_max=5, da_max=5, dh_max=5, endA=True, endB=True)
    writeMeshToGDF(vertices, panels)
    savedNodes, savedPanels = meshRectangularMember(stations, widths, heights, rA, rB, dz_max=2, da_max=2, dh_max=2, savedNodes=[], savedPanels=[], endA=True, endB=True)
    writeMesh(savedNodes, savedPanels, "Test")
