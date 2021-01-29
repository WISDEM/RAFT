import os
import sys
import subprocess as sub
import numpy as np

def nemohmesh_to_pnl(nemohMeshPath, oDir=None):
    '''
    convert mesh from .nemoh format to HAMS .pnl format

    Parameters
    ----------
    nemohMeshPath: str
        path to the nemoh mesh file.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Nemoh mesh must have:
      - single line header
      - line beginning with '0' separating points/panels
      - line beginning with '0' at the end of the file
      - no duplicate points (nodes)
    '''

    nemohMeshPath = os.path.normpath(nemohMeshPath)
    nemohDirName, nemohFileName = os.path.split(nemohMeshPath)
    if oDir is None:
        oDir = nemohDirName
    if os.path.isdir(oDir) is not True:
        os.makedirs(oDir)

    # N.B. Nemoh input files can have slightly different headers (panels and
    # points on top line...or just '2 0' or '2 1' on top line).
    iFile = open(nemohMeshPath, 'r')
    lines = iFile.readlines()
    header = lines[0].split()
    if header[0] == '2':
        ySym = int(header[1])
    else:
        ySym = 0
    ixOnes = []
    ixZeros = []
    for ix, line in enumerate(lines):
        if line.split()[0] == '0':
            ixZeros.append(ix)
        if line.split()[0] == '1':
            ixOnes.append(ix)

    numHeaders = ixOnes[0]
    numVertices = ixZeros[0] - numHeaders
    numPanels = ixZeros[1] - 1 - numVertices - numHeaders

    oFilePath = os.path.join(oDir, f'HullMesh.pnl')

    oFile = open(oFilePath, 'w')
    oFile.write(f'    --------------Hull Mesh File---------------\n\n')
    oFile.write(f'    # Number of Panels, Nodes, X-Symmetry and Y-Symmetry\n')
    oFile.write(f'         {numPanels}         {numVertices}         0         {ySym}\n\n')
    oFile.write(f'    #Start Definition of Node Coordinates     ! node_number   x   y   z\n')

    for ix, line in enumerate(lines[numHeaders:]):
        if line.split()[0] == '0':
            oFile.write(f'   #End Definition of Node Coordinates\n\n')
            oFile.write(f'   #Start Definition of Node Relations   ! panel_number  number_of_vertices   Vertex1_ID   Vertex2_ID   Vertex3_ID   (Vertex4_ID)\n')
            break
        oFile.write(f'{line.split()[0]:>5}{line.split()[1]:>18}{line.split()[2]:>18}{line.split()[3]:>18}\n')
    for ix, line in enumerate(lines[(numVertices+2):]):
        if line.split()[0] == '0':
            oFile.write(f'   #End Definition of Node Relations\n\n')
            oFile.write(f'    --------------End Hull Mesh File---------------\n')
            break
        if line.split()[0] == line.split()[3]:
            numNodes = 3
            oFile.write(f'{(ix+1):>5}{numNodes:>5}{line.split()[0]:>10}{line.split()[1]:>10}{line.split()[2]:>10}\n')
        else:
            numNodes = 4
            oFile.write(f'{(ix+1):>5}{numNodes:>5}{line.split()[0]:>10}{line.split()[1]:>10}{line.split()[2]:>10}{line.split()[3]:>10}\n')
    oFile.close()


def create_hams_dirs(baseDir=None):
    '''
    create necessary HAMS directories in baseDir

    Parameters
    ----------
    baseDir: str
        The top directory in which to create HAMS Input and Output directories

    Returns
    -------
    None

    Raises
    ------

    Notes
    -----

    '''

    if baseDir is None:
        baseDir = os.getcwd()
    else:
        baseDir = os.path.normpath(baseDir)
        if os.path.isdir(baseDir) is not True:
            os.makedirs(baseDir)

    inputDir = os.path.join(baseDir, f'Input')
    outputDirHams = os.path.join(baseDir, f'Output/Hams_format')
    outputDirHydrostar = os.path.join(baseDir, f'Output/Hydrostar_format')
    outputDirWamit = os.path.join(baseDir, f'Output/Wamit_format')

    if os.path.isdir(inputDir) is not True:
        os.mkdir(inputDir)
    if os.path.isdir(outputDirHams) is not True:
        os.makedirs(outputDirHams)
    if os.path.isdir(outputDirHydrostar) is not True:
        os.makedirs(outputDirHydrostar)
    if os.path.isdir(outputDirWamit) is not True:
        os.makedirs(outputDirWamit)

def write_hydrostatic_file(oDir=None, cog=np.zeros(3), mass=np.zeros((6,6)),
                           damping=np.zeros((6,6)), kHydro=np.zeros((6,6)),
                           kExt=np.zeros((6,6))):
    '''
    Writes Hydrostatic.in for HAMS (optional)

    Parameters
    ----------
    oDir: str
        directory to save Hydrostatic.in
    cog: array
        3x1 array - body's CoG
    mass: array
        6x6 array - body's mass matrix
    damping: array
        6x6 array - body's external damping matrix (i.e. non-radiation damping)
    kHydro: array
        6x6 array - body's hydrostatic stiffness matrix
    kExt: array
        6x6 array - body's additional stiffness matrix

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    '''
    if oDir is None:
        oDir = os.getcwd()
    else:
        oDir = os.path.normpath(oDir)

    f = open(os.path.join(oDir, 'Hydrostatic.in'), 'w')
    f.write(f' Center of Gravity:\n ')
    f.write(f'  {cog[0]:10.15E}  {cog[0]:10.15E}  {cog[0]:10.15E} \n')
    f.write(f' Body Mass Matrix:\n')
    for i in range(6):
        for j in range(6):
            f.write((f'   {mass[i,j]:10.5E}'))
        f.write(f'\n')
    f.write(f' External Damping Matrix:\n')
    for i in range(6):
        for j in range(6):
            f.write((f'   {damping[i,j]:10.5E}'))
        f.write(f'\n')
    f.write(f' Hydrostatic Restoring Matrix:\n')
    for i in range(6):
        for j in range(6):
            f.write((f'   {kHydro[i,j]:10.5E}'))
        f.write(f'\n')
    f.write(f' External Restoring Matrix:\n')
    for i in range(6):
        for j in range(6):
            f.write((f'   {kExt[i,j]:10.5E}'))
        f.write(f'\n')
    f.close()

def write_control_file(oDir=None, waterDepth=-50.0, iFType=3, oFType=3, numFreqs=-300,
                       minFreq=0.02, dFreq=0.02, freqList=None, numHeadings=1,
                       minHeading=0.0, dHeading=0.0,
                       refBodyCenter=[0.0, 0.0, 0.0], refBodyLen=1.0, irr=0,
                       numThreads=8):
    '''
    Description of the function

    Parameters
    ----------
    waterDepth : float
        water depth (m)
    iFType: int
        input frequency type
            1: deepwater wavenumber
            2: finite-depth wavenumber
            3: wave frequency
            4: wave period
            5: wavelength
    oFType: int
        output frequency type
            1: deepwater wavenumber
            2: finite-depth wavenumber
            3: wave frequency
            4: wave period
            5: wavelength
    numFreqs: int
        number of frequencies
            -ve: define minimum frequency and frequency step
            +ve: provide a list of frequencies
    minFreq: float
        minimum frequency (required if numFreqs is -ve)
    dFreq: float
        frequency step (required if numFreqs is -ve)
    freqList: list
        list of frequencies
    numHeadings: int
        number of headings
    minHeading: float
        minimum heading value (degs)
    dHeading: float
        heading step (degs)
    refBodyCenter: list
        reference body center
    refBodyLen: float
        reference body length
    irr: int
        irregular frequency removal option
    numThreads: int
        number of threads

    Returns
    -------

    Raises
    ------

    Notes
    -----

    '''

    if oDir is None:
        oDir = os.path.join(os.getcwd(), f'Input')
    else:
        oDir = os.path.normpath(oDir)

    oFileName = f'ControlFile.in'

    f = open(os.path.join(oDir, oFileName), 'w')
    f.write(f'   --------------HAMS Control file---------------\n\n')
    f.write(f'   Waterdepth  {waterDepth}D0\n\n')
    f.write(f'   #Start Definition of Wave Frequencies\n')
    f.write(f'    Input_frequency_type    {iFType}\n')
    f.write(f'    Output_frequency_type   {oFType}\n')
    f.write(f'    Number_of_frequencies   {numFreqs}\n') # -ve for min & step, +ve for list of frequencies (or periods)
    f.write(f'    Minimum_frequency_Wmin  {minFreq}D0\n')
    f.write(f'    Frequency_step          {dFreq}D0\n')
    f.write(f'   #End Definition of Wave Frequencies\n\n')
    f.write(f'   #Start Definition of Wave Headings\n')
    f.write(f'    Number_of_headings      -{numHeadings}\n')
    f.write(f'    Minimum_heading         {minHeading}D0\n')
    f.write(f'    Heading_step            {dHeading}D0\n')
    f.write(f'   #End Definition of Wave Headings\n\n')
    f.write(f'    Reference_body_center   {refBodyCenter[0]:.3f} {refBodyCenter[1]:.3f} {refBodyCenter[2]:.3f}\n')
    f.write(f'    Reference_body_length   {refBodyLen}D0\n')
    f.write(f'    If_remove_irr_freq      {irr}\n')
    f.write(f'    Number of threads       {numThreads}\n\n')
    f.write(f'   #Start Definition of Pressure and/or Elevation\n')
    f.write(f'    Number_of_field_points  0 \n')
    f.write(f'   #End Definition of Pressure and/or Elevation\n\n')
    f.write(f'   ----------End HAMS Control file---------------\n')
    f.close()


def read_wamit1(pathWamit1):
    '''
    Read added mass and damping from .1 file (WAMIT format)

    Parameters
    ----------
    pathWamit1 : f-str
        path to .1 file

    Returns
    -------
    addedMass : array
        added mass coefficients (nw x ndof x ndof)
    damping : array
        damping coefficients (nw x ndof x ndof)

    Raises
    ------

    Notes
    -----
    '''

    wamit1 = np.loadtxt(pathWamit1)
    w = np.unique(wamit1[:,0])
    addedMassCol = wamit1[:,3]
    dampingCol = wamit1[:,4]
    addedMass = addedMassCol.reshape((len(w)), 6, 6)
    damping = dampingCol.reshape((len(w), 6, 6))

    return addedMass, damping


def read_wamit3(pathWamit3):
    '''
    Read excitation force coefficients from .3 file (WAMIT format)

    Parameters
    ----------
    pathWamit3 : f-str
        path to .3 file

    Returns
    -------
    mod:

    phase:

    real:

    imag:

    Raises
    ------

    Notes
    -----
    '''

    wamit3 = np.loadtxt(pathWamit3)
    w = np.unique(wamit3[:,0])
    headings = np.unique(wamit3[:,1])
    mod = wamit3[:,3].reshape((len(w), len(headings), 6))
    phase = wamit3[:,4].reshape((len(w), len(headings), 6))
    real = wamit3[:,5].reshape((len(w), len(headings), 6))
    imag = wamit3[:,6].reshape((len(w), len(headings), 6))

    return mod, phase, real, imag

