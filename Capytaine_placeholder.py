
import numpy as np
import xarray as xr
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import capytaine as cpt


def interp_hydro_data(wCapy, wDes, addedMass, damping, fEx):
    addedMassInterp = spi.interp1d(wCapy, addedMass, axis=2)(wDes)
    dampingInterp = spi.interp1d(wCapy, damping, axis=2)(wDes)
    fExcitationInterp = spi.interp1d(wCapy, fEx)(wDes)
    return addedMassInterp, dampingInterp, fExcitationInterp


def read_capy_nc(capyFName, wDes=None):
    '''
    Read Capytaine .nc file and return hydrodynamic coefficients in suitable
    format for FrequencyDomain.py

    Parameters
    ----------
    capyFName : str
        String containing path to desired capytaine .nc file
    wDes: array
        Array of desired frequency points - can be different resolution to that
        used by Capytaine (but must be within the range computed by Capytaine)

    Returns
    -------
    frequency range & hydrodynamic coefficients. If wDes passed as argument,
    hydrodynamic coefficients are interpolated for these points (and frequency
    range is not returned)

    Raises
    ------
    ValueError
        when max/min wDes frequency range is outside wCapy range computed by
        Capytaine
    '''

    # read the Capytaine NetCDF data into xarray
    capyData = xr.open_dataset(capyFName)
    wCapy = capyData['omega'].values

    # check desired frequency range is within computed range (for interpolation)
    if wDes is not None:
        if wDes[0]<wCapy[0]:
            raise ValueError(f'\nMinimum desired frequency = {wDes[0]:.2f} rad/s \n'
                             f'Range of computed frequencies = {wCapy[0]:.2f} - {wCapy[-1]:.2f} rad/s \n'
                             f'[out of range]')
        if wDes[-1]>wCapy[-1]:
            raise ValueError(f'\nMaximum desired frequency = {wDes[-1]:.2f} rad/s \n'
                             f'Range of computed frequencies = {wCapy[0]:.2f} - {wCapy[-1]:.2f} rad/s \n'
                             f'[out of range]')

    # convert hydrodynamic coefficients to FrequencyDomain.py format
    addedMass = capyData['added_mass'].values.transpose(1,2,0)
    damping = capyData['radiation_damping'].values.transpose(1,2,0)
    fEx = (np.squeeze(capyData['diffraction_force'].values[0,:].T)
           + 1j*np.squeeze(capyData['diffraction_force'].values[1,:].T))

    # (optional) interpolate coefficients; return appropriate hydrodynamic
    # coefficients 
    if wDes is not None:
        addedMassIntp, dampingIntp, fExIntp = interp_hydro_data(wCapy,
                                                                wDes,
                                                                addedMass,
                                                                damping,
                                                                fEx)
        return wDes, addedMassIntp, dampingIntp, fExIntp
    else:
        return wCapy, addedMass, damping, fEx


def call_capy(meshFName, wCapy, CoG=[0,0,0], headings=[0.0], saveNc=False,
              ncFName=None, wDes=None):
    '''
    call Capytaine for a given mesh, frequency range and wave headings

    Parameters
    ----------
    meshFName : str
        string containing path to hydrodynamic mesh.
        mesh must be cropped at waterline (OXY plane) and have no lid
    wCapy: array
        array of frequency points to be computed by Capytaine
    CoG: list
        3x1 vector of body's CoG
    headings: list
        list of wave headings to compute
    saveNc: Bool
        save results to .nc file
    ncFName: str
        name of .nc file
    wDes: array
        array of desired frequency points
        (for interpolation of wCapy-based Capytaine data)

    Returns
    -------
    hydrodynamic coefficients; as computed or interpolated

    Notes
    -----
    TODO:
    - expand to multibody problems
    '''

    # create capytaine body object
    body = cpt.FloatingBody.from_file(meshFName)
    body.add_all_rigid_body_dofs()
    body.center_of_mass = CoG

    # define the hydrodynamic problems
    problems = [cpt.RadiationProblem(body=body,
                                     radiating_dof=dof,
                                     omega=w,
                                     sea_bottom=-np.infty,
                                     g=9.81,
                                     rho=1025)
                                     for dof in body.dofs for w in wCapy]

    problems += [cpt.DiffractionProblem(body=body,
                                        omega=w,
                                        wave_direction=heading,
                                        sea_bottom=-np.infty,
                                        g=9.81,
                                        rho=1025)
                                        for w in wCapy for heading in headings]

    # call Capytaine solver
    print(f'\n-------------------------------\n'
          f'Calling Capytaine BEM solver...\n'
          f'-------------------------------\n'
          f'mesh = {meshFName}\n'
          f'w range = {wCapy[0]:.3f} - {wCapy[-1]:.3f} rad/s\n'
          f'dw = {(wCapy[1]-wCapy[0]):.3f} rad/s\n'
          f'no of headings = {len(headings)}\n'
          f'no of radiation & diffraction problems = {len(problems)}\n'
          f'-------------------------------\n')

    solver = cpt.BEMSolver()
    results = [solver.solve(problem) for problem in sorted(problems)]
    capyData = cpt.assemble_dataset(results)

    # convert hydrodynamic coefficients to FrequencyDomain.py format
    addedMass = capyData['added_mass'].values.transpose(1,2,0)
    damping = capyData['radiation_damping'].values.transpose(1,2,0)
    fEx = np.squeeze(capyData['diffraction_force']).values.T

    # (optional) save to .nc file and interpolate coefficients; return
    # appropriate hydrodynamic coefficients
    if saveNc == True:
        cpt.io.xarray.separate_complex_values(capyData).to_netcdf(ncFName)

    if wDes is not None:
        addedMassIntp, dampingIntp, fExIntp = interp_hydro_data(wCapy,
                                                                wDes,
                                                                addedMass,
                                                                damping,
                                                                fEx)
        return wDes, addedMassIntp, dampingIntp, fExIntp
    else:
        return wCapy, addedMass, damping, fEx
