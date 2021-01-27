# This is a file to handle meshing of things in RAFT.
# Maybe we should use something more general instead, like meshBEM.

import numpy as np



def memberMesh(stations, diameters, rA, rB, dz_max=0, da_max=0):
    '''
    Creates mesh for an axisymmetric member as defined by RAFT.

    Parameters
    ----------
    stations:  list of locations along member axis at which the cross section will be specified
    diameters: list of corresponding diameters along member
    rA, rB: member end point coordinates
    dz_max: maximum panel height
    da_max: maximum panel width (before doubling azimuthal discretization)

    Returns
    -------
    vertices : array
        An array containing the mesh point coordinates, size [3, 4*npanel]
    '''

    
    radii = 0.5*np.array(diameters)


    # discretization defaults
    if dz_max==0:
        dz_max = stations[-1]/20
    if da_max==0:
        da_max = np.max(radii)/8
        

    # ------------------ discretize radius profile according to dz_max --------

    # radius profile data is contained in r_rp and z_rp
    r_rp = [radii[0]]
    z_rp = [0.0]

    # step through each station and subdivide as needed
    for i_s in range(1, len(radii)):
        dr_s = radii[i_s] - radii[i_s-1]; # delta r
        dz_s = stations[ i_s] - stations[ i_s-1]; # delta z
        # subdivision size
        if dr_s == 0: # vertical case
            cos_m=1
            sin_m=0
            dz_ps = dz_max; # (dz_ps is longitudinal dimension of panel)
        elif dz_s == 0: # horizontal case
            cos_m=0
            sin_m=1
            dz_ps = 0.6*da_max
        else: # angled case - set panel size as weighted average based on slope
            m = dr_s/dz_s; # slope = dr/dz
            dz_ps = np.arctan(np.abs(m))*2/np.pi*0.6*da_max + np.arctan(abs(1/m))*2/np.pi*dz_max;
            cos_m = dz_s/np.sqrt(dr_s**2 + dz_s**2)
            sin_m = dr_s/np.sqrt(dr_s**2 + dz_s**2)
        # make subdivision
        # local panel longitudinal discretization
        n_z = np.int(np.ceil( np.sqrt(dr_s*dr_s + dz_s*dz_s) / dz_ps ))
        # local panel longitudinal dimension
        d_l = np.sqrt(dr_s*dr_s + dz_s*dz_s)/n_z;
        for i_z in range(1,n_z+1):
            r_rp.append(  radii[i_s-1] + sin_m*i_z*d_l)
            z_rp.append(stations[i_s-1] + cos_m*i_z*d_l)
            
        print("-----")
        print(dz_s)
        print(d_l)


    # fill in end B if it's submerged
    n_r = np.int(np.ceil( radii[-1] / (0.6*da_max) ))   # local panel radial discretization #
    dr  = radii[-1] / n_r                               # local panel radial size

    for i_r in range(n_r):
        r_rp.append(radii[-1] - (1+i_r)*dr)
        z_rp.append(stations[-1])
    
    
    # fill in end A if it's submerged
    n_r = np.int(np.ceil( radii[0] / (0.6*da_max) ))   # local panel radial discretization #
    dr  = radii[0] / n_r                               # local panel radial size

    for i_r in range(n_r):
        r_rp.insert(0, radii[0] - (1+i_r)*dr)
        z_rp.insert(0, stations[0])
    
    
    # --------------- revolve radius profile, do adaptive paneling stuff ------

    # lists that we'll put the panel coordinates in, in lists of 4 coordinates per panel
    x = []
    y = []
    z = []

    npan =0;
    naz = np.int(8);

    # go through each point of the radius profile, panelizing from top to bottom:
    for i_rp in range(len(z_rp)-1):
    
        # rectangle coords - shape from outside is:  A D
        #                                            B C
        r1=r_rp[i_rp];
        r2=r_rp[i_rp+1];
        z1=z_rp[i_rp];
        z2=z_rp[i_rp+1];

        # scale up or down azimuthal discretization as needed
        while ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
            naz = np.int(2*naz)
        while ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
            naz = np.int(naz/2)

        # transition - increase azimuthal discretization
        if ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
            for ia in range(1, np.int(naz/2)+1):
                th1 = (ia-1  )*2*np.pi/naz*2;
                th2 = (ia-0.5)*2*np.pi/naz*2;
                th3 = (ia    )*2*np.pi/naz*2;

                x += [r1*np.cos(th1), r2*np.cos(th1), r2*np.cos(th2), (r1*np.cos(th1)+r1*np.cos(th3))/2 ]
                y += [r1*np.sin(th1), r2*np.sin(th1), r2*np.sin(th2), (r1*np.sin(th1)+r1*np.sin(th3))/2 ]
                z += [z1            , z2            , z2            , z1                                ]

                npan += 1

                x += [(r1*np.cos(th1)+r1*np.cos(th3))/2, r2*np.cos(th2), r2*np.cos(th3), r1*np.cos(th3)]
                y += [(r1*np.sin(th1)+r1*np.sin(th3))/2, r2*np.sin(th2), r2*np.sin(th3), r1*np.sin(th3)]
                z += [z1                               , z2            , z2            , z1            ]

                npan += 1

        # transition - decrease azimuthal discretization
        elif ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
            for ia in range(1, np.int(naz/2)+1):
                th1 = (ia-1  )*2*np.pi/naz*2;
                th2 = (ia-0.5)*2*np.pi/naz*2;
                th3 = (ia    )*2*np.pi/naz*2;
                x += [r1*np.cos(th1), r2*np.cos(th1), r2*(np.cos(th1)+np.cos(th3))/2, r1*np.cos(th2)]
                y += [r1*np.sin(th1), r2*np.sin(th1), r2*(np.sin(th1)+np.sin(th3))/2, r1*np.sin(th2)]
                z += [z1            , z2            , z2                            , z1            ]

                npan += 1;

                x += [r1*np.cos(th2), r2*(np.cos(th1)+np.cos(th3))/2, r2*np.cos(th3), r1*np.cos(th3)]
                y += [r1*np.sin(th2), r2*(np.sin(th1)+np.sin(th3))/2, r2*np.sin(th3), r1*np.sin(th3)]
                z += [z1            , z2                            , z2            , z1            ]

                npan += 1

        # no transition
        else:
            for ia in range(1, naz+1):
                th1 = (ia-1)*2*np.pi/naz;
                th2 = (ia  )*2*np.pi/naz;
                x += [r1*np.cos(th1), r2*np.cos(th1), r2*np.cos(th2), r1*np.cos(th2)]
                y += [r1*np.sin(th1), r2*np.sin(th1), r2*np.sin(th2), r1*np.sin(th2)]
                z += [z1            , z2            , z2            , z1            ]

                npan += 1

    # ----- transform coordinates to reflect specified rA and rB values
    
    vertices = np.array([x, y, z])

    rAB = rB - rA                                               # displacement vector from end A to end B [m]
    
    beta = np.arctan2(rAB[1],rAB[0])                            # member incline heading from x axis
    phi  = np.arctan2(np.sqrt(rAB[0]**2 + rAB[1]**2), rAB[2])   # member incline angle from vertical
    
    # trig terms for Euler angles rotation based on beta, phi, and gamma
    s1 = np.sin(beta) 
    c1 = np.cos(beta)
    s2 = np.sin(phi) 
    c2 = np.cos(phi)
    s3 = np.sin(0.0) 
    c3 = np.cos(0.0)

    R = np.array([[ c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3,  c1*s2],
                  [ c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3,  s1*s2],
                  [   -c3*s2      ,      s2*s3     ,    c2 ]])  #Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    
    vertices2 = np.matmul( R, vertices ) + rA[:,None]
    
    
    # >>>> likely need a step here to stop the mesh at the waterplane <<<<
    
    
    return vertices2.T


if __name__ == "__main__":
    
    stations = [0, 30, 40, 50]
    diameters = [10, 10, 4, 4]
    
    rA = np.array([-20, 0,-10])
    rB = np.array([10, 0, 20])
    
    vertices = memberMesh(stations, diameters, rA, rB, dz_max=2, da_max=2)
    
    npan = int(vertices.shape[0]/4)

    f = open("member.gdf", "w")
    f.write('gdf mesh \n')
    f.write('1.0   9.8 \n')
    f.write('0, 0 \n')
    f.write(f'{npan}\n')

    for i in range(npan*4):
        f.write(f'{vertices[i,0]:>10.3f} {vertices[i,1]:>10.3f} {vertices[i,2]:>10.3f}\n')
    
    f.close()
