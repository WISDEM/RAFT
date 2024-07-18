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

def meshRectangularMember(stations, widths, heights, rA, rB, dz_max=0, dw_max=0, dh_max=0, savedNodes=[], savedPanels=[]):
    '''
    Creates mesh for a rectangular member.

    Parameters
    ----------
    stations:  list 
        locations along member axis at which the cross section will be specified
    widths: list 
        corresponding widths along member
    heights: list
        corresponding heights along member
    rA, rB: list
        member end point coordinates
    dz_max: float
        maximum panel height
    dw_max: float
        maximum panel width
    dh_max: float
        maximum panel height
    savedNodes : list of lists
        all the node coordinates already saved
    savedPanels : list
        the information for all the panels already saved: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID   Vertex3_ID   (Vertex4_ID)

    Returns
    -------
    nodes : list
        list of node coordinates
    panels : list
        the information for all the panels: panel_number number_of_vertices   Vertex1_ID   Vertex2_ID   Vertex3_ID   (Vertex4_ID)
    '''

    # discretization defaults
    if dz_max == 0:
        dz_max = stations[-1] / 20
    if dw_max == 0:
        dw_max = np.max(widths) / 8
    if dh_max == 0:
        dh_max = np.max(heights) / 8

    # calculate member rotation matrix
    rAB = rB - rA  # displacement vector from end A to end B [m]
    beta = np.arctan2(rAB[1], rAB[0])  # member incline heading from x axis
    phi = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])  # member incline angle from vertical

    s1 = np.sin(beta)
    c1 = np.cos(beta)
    s2 = np.sin(phi)
    c2 = np.cos(phi)
    s3 = np.sin(0.0)
    c3 = np.cos(0.0)

    R = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                  [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                  [-c3 * s2, s2 * s3, c2]])  # Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    nSavedPanelsOld = len(savedPanels)

    # Create intermediate stations based on dz_max
    new_stations = [stations[0]]
    new_widths = [widths[0]]
    new_heights = [heights[0]]
    for i in range(len(stations) - 1):
        z1, z2 = stations[i], stations[i + 1]
        w1, w2 = widths[i], widths[i + 1]
        h1, h2 = heights[i], heights[i + 1]

        dz = z2 - z1
        num_divisions = max(int(np.ceil(dz / dz_max)), 1)
        z_divisions = np.linspace(z1, z2, num_divisions + 1)
        for j in range(1, num_divisions + 1):
            new_stations.append(z_divisions[j])
            new_widths.append(w1 + (w2 - w1) * (z_divisions[j] - z1) / dz)
            new_heights.append(h1 + (h2 - h1) * (z_divisions[j] - z1) / dz)

    # Generate panels for all divisions
    for i in range(len(new_stations) - 1):
        w1, w2 = new_widths[i], new_widths[i + 1]
        h1, h2 = new_heights[i], new_heights[i + 1]
        z1, z2 = new_stations[i], new_stations[i + 1]

        nw = max(int(np.ceil(w1 / dw_max)), int(np.ceil(w2 / dw_max)))
        nh = max(int(np.ceil(h1 / dh_max)), int(np.ceil(h2 / dh_max)))

        x_divisions_w1 = np.linspace(-w1 / 2, w1 / 2, nw + 1)
        x_divisions_w2 = np.linspace(-w2 / 2, w2 / 2, nw + 1)
        y_divisions_h1 = np.linspace(-h1 / 2, h1 / 2, nh + 1)
        y_divisions_h2 = np.linspace(-h2 / 2, h2 / 2, nh + 1)

        for iw in range(nw):
            for ih in range(nh):
                # Panel 1: Front face
                X = [x_divisions_w1[iw], x_divisions_w1[iw + 1], x_divisions_w2[iw + 1], x_divisions_w2[iw]]
                Y = [-h1 / 2, -h1 / 2, -h2 / 2, -h2 / 2]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

                # Panel 2: Back face
                X = [x_divisions_w1[iw], x_divisions_w1[iw + 1], x_divisions_w2[iw + 1], x_divisions_w2[iw]]
                Y = [h1 / 2, h1 / 2, h2 / 2, h2 / 2]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

        for iw in range(nw):
            for ih in range(nh):
                # Panel 3: Left face
                X = [-w1 / 2, -w1 / 2, -w2 / 2, -w2 / 2]
                Y = [y_divisions_h1[ih], y_divisions_h1[ih + 1], y_divisions_h2[ih + 1], y_divisions_h2[ih]]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

                # Panel 4: Right face
                X = [w1 / 2, w1 / 2, w2 / 2, w2 / 2]
                Y = [y_divisions_h1[ih], y_divisions_h1[ih + 1], y_divisions_h2[ih + 1], y_divisions_h2[ih]]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

    # Mesh the end faces
    wA = new_widths[0]
    hA = new_heights[0]
    wB = new_widths[-1]
    hB = new_heights[-1]
    zA = new_stations[0]
    zB = new_stations[-1]

    # End A face
    nwA = max(int(np.ceil(wA / dw_max)), 1)
    nhA = max(int(np.ceil(hA / dh_max)), 1)
    x_divisions_wA = np.linspace(-wA / 2, wA / 2, nwA + 1)
    y_divisions_hA = np.linspace(-hA / 2, hA / 2, nhA + 1)

    for iw in range(nwA):
        for ih in range(nhA):
            X = [x_divisions_wA[iw], x_divisions_wA[iw + 1], x_divisions_wA[iw + 1], x_divisions_wA[iw]]
            Y = [y_divisions_hA[ih], y_divisions_hA[ih], y_divisions_hA[ih + 1], y_divisions_hA[ih + 1]]
            Z = [zA, zA, zA, zA]
            nodes0 = np.array([X, Y, Z])
            nodes = np.matmul(R, nodes0) + rA[:, None]
            makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

    # End B face
    nwB = max(int(np.ceil(wB / dw_max)), 1)
    nhB = max(int(np.ceil(hB / dh_max)), 1)
    x_divisions_wB = np.linspace(-wB / 2, wB / 2, nwB + 1)
    y_divisions_hB = np.linspace(-hB / 2, hB / 2, nhB + 1)

    for iw in range(nwB):
        for ih in range(nhB):
            X = [x_divisions_wB[iw], x_divisions_wB[iw + 1], x_divisions_wB[iw + 1], x_divisions_wB[iw]]
            Y = [y_divisions_hB[ih], y_divisions_hB[ih], y_divisions_hB[ih + 1], y_divisions_hB[ih + 1]]
            Z = [zB, zB, zB, zB]
            nodes0 = np.array([X, Y, Z])
            nodes = np.matmul(R, nodes0) + rA[:, None]
            makepanel(nodes[0], nodes[1], nodes[2], savedNodes, savedPanels)

    print(f'Of {len(savedPanels) - nSavedPanelsOld} generated panels, {len(savedPanels) - nSavedPanelsOld} were submerged and have been used in the mesh.')

    return savedNodes, savedPanels

def writeMesh(savedNodes, savedPanels, oDir=""):
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


def meshRectangularMemberForGDF(stations, widths, heights, rA, rB, dz_max=0, dw_max=0, dh_max=0, endA=True, endB=True):
    '''
    Creates mesh for a rectangular member.

    Parameters
    ----------
    stations: list of locations along member axis at which the cross section will be specified
    widths: list of corresponding widths along member
    heights: list of corresponding heights along member
    rA, rB: member end point coordinates
    dz_max: maximum panel height
    dw_max: maximum panel width
    dh_max: maximum panel height
    endA/endB: flag for whether to mesh each end

    Returns
    -------
    vertices : array
        An array containing the mesh point coordinates, size [3, 4*npanel]
    '''

    if len(stations) != len(widths) or len(stations) != len(heights):
        raise ValueError("The lengths of stations, widths, and heights must be the same.")

    if dz_max == 0:
        dz_max = stations[-1] / 20
    if dw_max == 0:
        dw_max = max(widths) / 8
    if dh_max == 0:
        dh_max = max(heights) / 8

    rAB = np.array(rB) - np.array(rA)
    beta = np.arctan2(rAB[1], rAB[0])
    phi = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])

    s1 = np.sin(beta)
    c1 = np.cos(beta)
    s2 = np.sin(phi)
    c2 = np.cos(phi)
    s3 = np.sin(0.0)
    c3 = np.cos(0.0)

    R = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                  [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                  [-c3 * s2, s2 * s3, c2]])

    nSavedPanelsOld = 0

    new_stations = [stations[0]]
    new_widths = [widths[0]]
    new_heights = [heights[0]]
    for i in range(len(stations) - 1):
        z1, z2 = stations[i], stations[i + 1]
        w1, w2 = widths[i], widths[i + 1]
        h1, h2 = heights[i], heights[i + 1]

        dz = z2 - z1
        if dz == 0:
            continue

        num_divisions = max(int(np.ceil(dz / dz_max)), 1)
        z_divisions = np.linspace(z1, z2, num_divisions + 1)
        for j in range(1, num_divisions + 1):
            new_stations.append(z_divisions[j])
            new_widths.append(w1 + (w2 - w1) * (z_divisions[j] - z1) / dz)
            new_heights.append(h1 + (h2 - h1) * (z_divisions[j] - z1) / dz)

    x, y, z = [], [], []

    for i in range(len(new_stations) - 1):
        w1, w2 = new_widths[i], new_widths[i + 1]
        h1, h2 = new_heights[i], new_heights[i + 1]
        z1, z2 = new_stations[i], new_stations[i + 1]

        if w1 == 0 or w2 == 0 or h1 == 0 or h2 == 0:
            continue

        if dw_max == 0 or dh_max == 0:
            raise ValueError("dw_max and dh_max must be non-zero.")

        nw = max(int(np.ceil(w1 / dw_max)), int(np.ceil(w2 / dw_max)))
        nh = max(int(np.ceil(h1 / dh_max)), int(np.ceil(h2 / dh_max)))

        x_divisions_w1 = np.linspace(-w1 / 2, w1 / 2, nw + 1)
        x_divisions_w2 = np.linspace(-w2 / 2, w2 / 2, nw + 1)
        y_divisions_h1 = np.linspace(-h1 / 2, h1 / 2, nh + 1)
        y_divisions_h2 = np.linspace(-h2 / 2, h2 / 2, nh + 1)

        for iw in range(nw):
            for ih in range(nh):
                X = [x_divisions_w1[iw], x_divisions_w1[iw + 1], x_divisions_w2[iw + 1], x_divisions_w2[iw]]
                Y = [-h1 / 2, -h1 / 2, -h2 / 2, -h2 / 2]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

                X = [x_divisions_w1[iw], x_divisions_w1[iw + 1], x_divisions_w2[iw + 1], x_divisions_w2[iw]]
                Y = [h1 / 2, h1 / 2, h2 / 2, h2 / 2]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

        for iw in range(nw):
            for ih in range(nh):
                X = [-w1 / 2, -w1 / 2, -w2 / 2, -w2 / 2]
                Y = [y_divisions_h1[ih], y_divisions_h1[ih + 1], y_divisions_h2[ih + 1], y_divisions_h2[ih]]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

                X = [w1 / 2, w1 / 2, w2 / 2, w2 / 2]
                Y = [y_divisions_h1[ih], y_divisions_h1[ih + 1], y_divisions_h2[ih + 1], y_divisions_h2[ih]]
                Z = [z1, z1, z2, z2]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

    wA = new_widths[0]
    hA = new_heights[0]
    wB = new_widths[-1]
    hB = new_heights[-1]
    zA = new_stations[0]
    zB = new_stations[-1]

    if endA and wA > 0 and hA > 0:
        nwA = max(int(np.ceil(wA / dw_max)), 1)
        nhA = max(int(np.ceil(hA / dh_max)), 1)
        x_divisions_wA = np.linspace(-wA / 2, wA / 2, nwA + 1)
        y_divisions_hA = np.linspace(-hA / 2, hA / 2, nhA + 1)

        for iw in range(nwA):
            for ih in range(nhA):
                X = [x_divisions_wA[iw], x_divisions_wA[iw + 1], x_divisions_wA[iw + 1], x_divisions_wA[iw]]
                Y = [y_divisions_hA[ih], y_divisions_hA[ih], y_divisions_hA[ih + 1], y_divisions_hA[ih + 1]]
                Z = [zA, zA, zA, zA]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

    if endB and wB > 0 and hB > 0:
        nwB = max(int(np.ceil(wB / dw_max)), 1)
        nhB = max(int(np.ceil(hB / dh_max)), 1)
        x_divisions_wB = np.linspace(-wB / 2, wB / 2, nwB + 1)
        y_divisions_hB = np.linspace(-hB / 2, hB / 2, nhB + 1)

        for iw in range(nwB):
            for ih in range(nhB):
                X = [x_divisions_wB[iw], x_divisions_wB[iw + 1], x_divisions_wB[iw + 1], x_divisions_wB[iw]]
                Y = [y_divisions_hB[ih], y_divisions_hB[ih], y_divisions_hB[ih + 1], y_divisions_hB[ih + 1]]
                Z = [zB, zB, zB, zB]
                nodes0 = np.array([X, Y, Z])
                nodes = np.matmul(R, nodes0) + rA[:, None]
                x.extend(nodes[0])
                y.extend(nodes[1])
                z.extend(nodes[2])

    vertices = np.array([x, y, z])

    vertices2 = np.matmul(R, vertices) + rA[:, None]

    return vertices2.T

def writeMeshToGDF(vertices,  filename="platform.gdf", aboveWater=True):

    npan = int(vertices.shape[0]/4)

    f = open(filename, "w")
    f.write('gdf mesh \n')
    f.write('1.0   9.8 \n')
    f.write('0, 0 \n')
    f.write(f'{npan}\n')
    
    if aboveWater:
        for i in range(npan*4):
            f.write(f'{vertices[i,0]:>10.3f} {vertices[i,1]:>10.3f} {vertices[i,2]:>10.3f}\n')
            
    else: # this option avoids making panels above the waterline
    
        for i in range(npan):
            
            panel = vertices[4*i:4*i+4]  # the vertices of this panel
            
            if any(panel[:,2] < -0.001):  # only consider the panel if it's at least partly submerged (some z < 0)
                
                for j in range(4):   # go through each vertex of the panel, but move any above z=0 down to z=0
                
                    if panel[j,2] > 0:  panel[j,2] = 0
                    
                    f.write(f'{panel[j,0]:>10.3f} {panel[j,1]:>10.3f} {panel[j,2]:>10.3f}\n')
    
    f.close()    

if __name__ == "__main__":

    stations= [-120, -12,  -4,  10]
    widths = [24, 24, 24,  24]
    heights = [24, 24, 24,  24]
    
    rA = np.array([0, 0,-10])
    rB = np.array([0, 0, 22])


    vertices = meshRectangularMemberForGDF(stations, widths, heights, rA, rB, dz_max=20, dw_max=20, dh_max=20)
    writeMeshToGDF(vertices)
    savedNodes, savedPanels = meshMember(stations, widths, heights, rA, rB, dz_max=20, dw_max=20, dh_max=20)
    writeMesh(savedNodes, savedPanels, "Test")
