import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy import integrate
from scipy import optimize

constants = {'g': 9.81,
             'Rd' : 287.,
             'cpd' : 1004.,
             'Lv': 2.5e6
            }

def smooth(ds,sigma=2):
    """Gaussian smoothing of a xarray object"""
    return xr.apply_ufunc(gaussian_filter,ds,kwargs={"sigma":sigma})

def sel_cond(ds,cond):
    """
    Select values from ds where cond is True, and flatten the result into a 
    1D array, excluding NaN values.

    Parameters
    ----------
    ds : xarray.DataArray
        The data array from which to select values.
    cond : xarray.DataArray
        A boolean array of the same shape as ds, where True indicates the 
        values to select.
    Returns
    -------
    numpy.ndarray
        A 1D array of the selected values from ds, excluding NaN values.
    """
    ar = ds.where(cond).data.flatten()
    return ar[~np.isnan(ar)]

def get_hist(T1,T2,cond,dim_names = None,weights=None,bins=None):
    """Make a 2D histogram of T1 vs T2, selecting only values where cond is 
    True, and optionally using weights.

    Parameters
    ----------
    T1 : xarray.DataArray
        The first variable to be binned.
    T2 : xarray.DataArray
        The second variable to be binned.
    cond : xarray.DataArray
        A boolean array of the same shape as T1 and T2, where True indicates 
        the values to include in the histogram.
    dim_names : list of str, optional
        Names for the dimensions of the output histogram. If None, default names
        "dim_1" and "dim_2" will be used.
    weights : xarray.DataArray, optional
        An array of the same shape as T1 and T2, containing weights for each
        value. If None, all values will be weighted equally, i.e. each bin 
        will count the number of values that fall into it.
    bins : tuple of array-like, optional
        A tuple (bins_1, bins_2) where bins_1 and bins_2 are the bin edges 
        for T1 and T2, respectively. If None, bins will be automatically 
        determined based on the range of T1 and T2, with 100 bins for each 
        variable.

    Returns
    -------
    xarray.DataArray
        A 2D histogram of T1 vs T2 for values in cond and optionally weighted 
        by weights.
    """
    if dim_names is None:
        dim_names = ["dim_1","dim_2"]

    if bins is None:
        bins_1 = np.linspace(T1.min()-1,T1.max()+1,101)
        bins_2 = np.linspace(T2.min()-1,T2.max()+1,101)
    else:
        bins_1, bins_2 = bins

    T1 = sel_cond(T1,cond)
    T2 = sel_cond(T2,cond)
    weights = sel_cond(weights,cond) if weights is not None else None

    bins_1_center = (bins_1[:-1]+bins_1[1:])/2
    bins_2_center = (bins_2[:-1]+bins_2[1:])/2

    hist2d = np.histogram2d(T1, T2, weights=weights, bins=(bins_1, bins_2))[0]
    
    hist2d_xr = xr.DataArray(hist2d,
                             dims = dim_names,
                             coords = {dim_names[0]:bins_1_center,dim_names[1]:bins_2_center}
                             )
    return hist2d_xr

def humidsat(t,p):
    """computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    pressure (p) in hPa.
    
    These are all computed using the modified Tetens-like formulae given by
    Buck (1981, J. Appl. Meteorol.)
    for vapor pressure over liquid water at temperatures over 0 C, and for
    vapor pressure over ice at temperatures below -23 C, and a quadratic
    polynomial interpolation for intermediate temperatures.

    Parameters
    ----------
    t : array-like
        Temperature in K
    p : array-like
        Pressure in hPa

    Returns
    -------
    esat : array-like
        Saturation vapor pressure in hPa
    qsat : array-like
        Saturation specific humidity (dimensionless, kg/kg)
    rsat : array-like
        Saturation mixing ratio (dimensionless, kg/kg)
    """
    
    tc=t-273.16
    tice=-23
    t0=0
    Rd=287.04
    Rv=461.5
    epsilon=Rd/Rv


    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))*((tc-tice)/(t0-tice))

    esat=(tc<tice)*eice + (tc>t0)*ewat + (tc>tice)*(tc<t0)*eint

    #now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    
    return esat,qsat,rsat

def qsat(t,p):
    """Calculate saturation specific humidity using a modified Tetens-like 
    formula (Buck, 1981, J. Appl. Meteorol.).
    
    Parameters
    ----------
    t : array-like
        Temperature in K
    p : array-like
        Pressure in hPa

    Returns
    -------
    qsat : array-like
        Saturation specific humidity (dimensionless, kg/kg)
    """
    _,q,_=humidsat(t,p)
    return q

########################################################################
######################### VERTICAL INTERPOLATION #######################
########################################################################

def searchsorted_custom(a, v, side='left', axis=0):
    """Return insertion indices of ``v`` in ``a`` along a chosen axis.

    This is an extension of ``np.searchsorted`` that handles multi-dimensional 
    arrays where values are monotonic along a specified axis.

    Parameters
    ----------
    a : np.ndarray
        Input array (typically monotonic along ``axis``).
    v : float or np.ndarray
        Value(s) to locate in ``a``. Must be broadcast-compatible with ``a``
        after reduction along ``axis``.
    side : {'left', 'right'}, default 'left'
        If 'left', returns the first valid insertion index (uses ``a < v``).
        If 'right', returns the last valid insertion index (uses ``a <= v``).
    axis : int, default 0
        Axis along which to compute insertion indices.

    Returns
    -------
    np.ndarray
        Insertion index/indices with shape equal to ``a`` without ``axis``.
    """
    if side == 'left':
        return (a < v).sum(axis=axis)
    elif side == 'right':
        return (a <= v).sum(axis=axis)
    else:
        raise ValueError("side must be 'left' or 'right'")


def add_dim(ar):
    """Add a leading singleton dimension to an array-like object.

    Parameters
    ----------
    ar : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        View of ``ar`` with shape ``(1, ...)``.
    """
    return ar[np.newaxis, ...]


def pressure_itp_modellevs_numpy(ar, pressure, target_pressure):
    """Linearly interpolate a field from model levels to a target pressure.

    Interpolation is done along the first axis (vertical levels). For target
    pressures below the surface or above model top, the nearest edge value is
    returned (index clipping).

    Parameters
    ----------
    ar : np.ndarray
        Field values on model levels, with vertical dimension on axis 0.
    pressure : np.ndarray
        Pressure values (hPa) on model levels, same shape as ``ar`` along axis 0.
    target_pressure : float or np.ndarray
        Target pressure level(s) in hPa. Can be scalar or array broadcastable
        to horizontal/time dimensions.

    Returns
    -------
    np.ndarray
        Interpolated field at ``target_pressure`` with axis 0 removed.
    """
    ik = np.maximum(searchsorted_custom(pressure, target_pressure, side='right') - 1, 0)
    ikp1 = np.minimum(searchsorted_custom(pressure, target_pressure, side='right'), len(pressure) - 1)

    ik = add_dim(ik)
    ikp1 = add_dim(ikp1)

    pk = np.take_along_axis(pressure, ik, axis=0)[0]
    pkp1 = np.take_along_axis(pressure, ikp1, axis=0)[0]
    fk = np.take_along_axis(ar, ik, axis=0)[0]
    fkp1 = np.take_along_axis(ar, ikp1, axis=0)[0]

    pkp1 += (pkp1 == pk)  # avoid division by zero
    ar_itp = fk + (fkp1 - fk) * (target_pressure - pk) / (pkp1 - pk)

    return ar_itp


def pressure_itp_modellevs(da, pressure, target_pressure):
    """xarray wrapper for pressure interpolation on model levels.

    Uses :func:`pressure_itp_modellevs_numpy` on ``da.data[0]`` (i.e., assumes
    an extra leading dimension is present), then rebuilds an ``xr.DataArray``
    without the ``lev`` dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input variable containing a ``lev`` dimension.
    pressure : xr.DataArray
        Pressure field corresponding to ``da`` (hPa).
    target_pressure : float or xr.DataArray
        Target pressure level(s) in hPa.

    Returns
    -------
    xr.DataArray
        Interpolated field with same dimensions as ``da`` except ``lev`` removed.
    """
    if isinstance(target_pressure, xr.DataArray):
        target_pressure = target_pressure.values
    values = pressure_itp_modellevs_numpy(da.data[0], pressure.data, target_pressure)
    coords = dict(da.coords.copy())
    del coords['lev']
    return xr.DataArray(values[np.newaxis, ...], dims=[dim for dim in da.dims if dim != 'lev'], coords=coords)

########################################################################
############################## STATION DATA ############################
########################################################################

def calc_blh_richardson(sounding):
    """Calculate the PBL top pressure using a bulk Richardson number-based method, 
    from sounding data. The method is the same as that used in ERA5, with the lowest 
    sounding level used in the calculation of the Bulk Richardson number. Virtual 
    effects are neglected.

    Parameters
    ----------
    sounding: xarray.Dataset
        Dataset containing sounding data. Must have dimension 'lev' and variables 
        'Uspd' (m/s), 'T' (K), 'Z' (m), and 'pressure' (Pa). 'pressure' should be 
        ordered from surface to top (i.e. pressure decreases with increasing lev).

    Returns
    -------
    bltop_pres: xarray.DataArray
        Pressure at the top of the PBL, in hPa.
    """
    DeltaUsq = sounding.Uspd**2
    
    ghbl = constants['g'] * sounding.Z
    svhbl = constants['cpd'] * sounding.T + ghbl
    gzn = ghbl.isel(lev=0)
    svn = svhbl.isel(lev=0)

    Ri = (ghbl - gzn) * 2 * (svhbl - svn) / ( svhbl + svn - gzn - ghbl ) / DeltaUsq

    # Find lowest level where Ri > 0.25, then linearly interpolate to find the pressure where Ri = 0.25
    Ri = Ri.fillna(0.)
    lev1 = (Ri > 0.25).argmax('lev')
    lev2 = lev1 - 1
    p1 = sounding.pressure.isel(lev = lev1)
    p2 = sounding.pressure.isel(lev = lev2)
    Ri1 = Ri.isel(lev = lev1)
    Ri2 = Ri.isel(lev = lev2)
    bltop_pres = p1 + (p2 - p1) * ( 0.25 - Ri1 ) / (Ri2 - Ri1)
    
    # In some cases p2 is nan. Then, set bltop_pres to p1
    bltop_pres[np.isnan(bltop_pres)] = p1[np.isnan(bltop_pres)].values

    # when lev1 is 0, it means no level has Ri > 0.25, so set bltop_pres to nan
    bltop_pres = bltop_pres.where(lev1 > 0)

    return bltop_pres / 100

class Station():
    def __init__(self, name, lat, lon, path_soundings ):
        """Class to handle sounding data for a given station. It loads the IGRA and ERA5 data from 
        a specified path and calculates some additional variables such as potential temperature, 
        PBL height based on the Richardson number, and the superadiabatic layer strength.

        Parameters
        ----------
        name : str
            Name of the station.
        lat : float
            Latitude of the station.
        lon : float
            Longitude of the station.
        path_soundings : str
            Path to the sounding data files.
        """
        self.name = name
        self.lat = lat
        self.lon = lon
        self.path_soundings = path_soundings
        self.get_igra_data()
        self.get_era5_data()
        self.era5_where_igra_valid = self.era5.sel(time = np.isin(self.era5.time,self.igra.time))
        
    def get_igra_data(self):
        self.igra = xr.open_dataset(f'{self.path_soundings}/{self.name}_IGRA_JJA_2001-2021.nc')
        self.igra['theta'] = self.igra.T * (self.igra.pressure / 1e5 )**(-constants['Rd']/constants['cpd'])
        self.igra['PBL_HPA_richardson'] = calc_blh_richardson(self.igra)
        T500 = pressure_itp_modellevs(self.igra.T[:,::-1].expand_dims(dummy=1).transpose('dummy','lev','time'), 
                                      self.igra.pressure[:,::-1].fillna(1.).transpose('lev','time')/100, 
                                      500)
        self.igra['T500'] = T500.isel(dummy=0)
        self.igra['superadibatic_strength'] = self.igra.theta.isel(lev = 0) - self.igra.theta.fillna(1e3).min('lev')

    def get_era5_data(self):
        self.era5 = xr.open_dataset(f'{self.path_soundings}/{self.name}_ERA5_JJA_2001-2021_00UTC.nc')
        self.era5['superadibatic_strength_2m'] = np.maximum(0.,self.era5.THETA2M - self.era5.THETA.min('lev'))
        

########################################################################
######################## MSE-CONSERVING PROFILES #######################
########################################################################

def MSEstar(T,z,p):
    """Calculates saturation moist static energy.
    T in K, z in m, p in hPa"""
    return constants['cpd'] * T + constants['g'] * z + constants['Lv'] * qsat(T, p)  # MSE_star in J/kg

def iterate_constant_MSEstar(MSE0,z,p_init,T_init):
    """Given a reference MSE^* value, an array of heights, and a guess 
    for the pressure and temperature profile, this function calculates the temperature profile
    that satisfies the condition cp * T + g * z * qsat(T,p_init) = MSE0.
    It also calculates the corresponding hydrostatic pressure profile.

    Parameters
    ----------
    MSE0 : float
        Reference MSE^* value in J/kg.
    z : array-like
        Heights in meters.
    p_init : array-like
        Initial guess for the pressure profile in hPa; should have the same length as z.
    T_init : array-like
        Initial guess for the temperature profile in Kelvin; used to initialize the optimizer.

    Returns
    -------
    p_new : array-like
        New hydrostatic pressure profile in hPa; satisfies d(p_new)/dz = - p_new * g / (R*T_new).
    T_new : array-like
        New temperature profile in Kelvin that satisfies the MSE^* condition.
    """
    obj_func = lambda T,i: ((MSEstar(T,z[i],p_init[i]) - MSE0)**2) # Objective function to minimize
    T_new = np.array([optimize.minimize(lambda T: obj_func(T,i), T_init[i]).x[0] for i in range(len(T_init))])
    lnp_new = integrate.cumulative_trapezoid(-constants['g'] / (constants['Rd'] * T_new), z, initial=0)
    return np.exp(lnp_new) * 1e3 , T_new
   
def calc_moist_adiabat(MSE0):
    """Calculates a moist adiabat, i.e. a profile that satisfies the condition cp * T + g * z * qsat(T,p) = MSE0,
    along with hydrostasy. This is calculated iteratively: an initial guess for the pressure profile is made,
    then temperature is inverted from the MSE^* condition, the pressure profile is updated to satisfy hydrostatic balance,
    etc. until convergence.
    
    Parameters
    ----------
    MSE0 : float
        Reference MSE^* value in J/kg, which is the target for the moist adiabat.

    Returns
    -------
    z : array-like
        Heights in meters.
    pz : array-like
        Hydrostatic pressure profile in hPa.
    Tz : array-like
        Moist adiabatic temperature profile in Kelvin.
    
    """
    z = np.linspace(0,20e3,401)

    # Initial guess: dry adiabat
    Tz = (MSE0 - constants['g'] * z) /constants['cpd']
    lnp = integrate.cumulative_trapezoid(- constants['g'] / (constants['Rd'] * Tz), z, initial=0)
    pz = np.exp(lnp) * 1e3
    T_new = Tz; Tz = Tz + 1  # just to enter the loop
    while ((T_new - Tz)**2).mean() > 1e-6: # until convergence
        Tz = T_new
        pz, T_new = iterate_constant_MSEstar(MSE0, z, pz, Tz)
    return z,pz,Tz

def calc_parcel_profile(T0, MSE0, ma_profile = None):
    """
    Calculates the profile of an air parcel lifted from 
    a surface at zero elevation, pressure 1000 hPa, temperature T0, and
    moist static energy MSE0, and conserving its MSE. The pressure profile is hydrostatic 
    for a reference temperature profile that is moist adiabatic with MSE^* = MSE0.
    Note that the parcel's pressure is thus generally not hydrostatic.
    The parcel is lifted dry adiabatically until it reaches saturation, 
    then moist adiabatically.

    Parameters
    ----------
    T0 : float
        Surface temperature in Kelvin.
    MSE0 : float
        Parcel's conserved MSE value in J/kg.
    ma_profile : tuple of array-like, optional
        If provided, the moist adiabat profile to use for the reference temperature profile. 
        Should be a tuple (z, pz, Tz) of arrays of the same length, giving the height, 
        pressure and temperature profiles of a moist adiabat with MSE^* = MSE0. 
        If None (default), the moist adiabat profile is calculated using the function 
        calc_moist_adiabat.
    """
    q0 = (MSE0 - constants['cpd'] * T0) / constants['Lv']
    print(f'Initial q0: {q0:.4f} kg/kg')

    if ma_profile is None:
        z,pz,Tz = calc_moist_adiabat(MSE0)
    else:
        z,pz,Tz = ma_profile

    Tparc = T0 - constants['g'] / constants['cpd'] * z
    i_sat = np.argmax(Tparc<Tz)
    Tparc[i_sat:] = Tz[i_sat:]
    return pz,Tparc,i_sat
    