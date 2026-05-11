import metpy.calc as mpcalc
from metpy.plots import  SkewT
from metpy.units import units
import numpy as np

def plot_skewT(fig,gs_element,p,T,q,ps,T2m,Td2m,dpcolor='g',alpha_refprof=0.15):
    """Plot a Skew-T diagram for given pressure (p), temperature (T), and specific humidity (q) profiles.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to plot on.
    gs_element : gridspec.GridSpec
        The subplot using matplotlib.gridspec
    p: np.array
        Pressure profile in hPa.
    T: np.array
        Temperature profile in K.
    q: np.array
        Specific humidity profile in kg/kg.
    ps: float
        Surface pressure in hPa.
    T2m: float
        2-m temperature in K.
    Td2m: float
        2-m dew-point temperature in K.
    dpcolor: str, optional
        Color for the dewpoint temperature line.
    alpha_refprof: float, optional
        Alpha value for the reference profiles.
    """
    Td = mpcalc.dewpoint_from_specific_humidity(p * units.hPa, q )
    
    skew = SkewT(fig=fig,subplot=gs_element, rotation=45,aspect=80.5)
    
    skew.plot(p * units.hPa, T * units.degK, 'r',linewidth=3)
    skew.plot(p * units.hPa, Td, dpcolor,linewidth=3)

    skew.plot(ps * units.hPa, T2m * units.degK, 'o', color='w', markerfacecolor='r')
    skew.plot(ps * units.hPa, Td2m * units.degK, 'o', color='w', markerfacecolor='g')

    # Calculate full parcel profile, LCL, and CAPE area
    full_pprof = np.concatenate([p,[ps]]) * units.hPa
    full_Tprof = np.concatenate([T,[T2m]]) * units.degK
    D2M_withunits = Td2m * units.degK

    lcl_pressure, lcl_temperature = mpcalc.lcl(full_pprof[-1], full_Tprof[-1], D2M_withunits)
    prof = mpcalc.parcel_profile(full_pprof[::-1], full_Tprof[-1], D2M_withunits).to('degC')[::-1]
    skew.plot(full_pprof, prof, 'orange', linewidth=2)
    skew.plot(lcl_pressure, lcl_temperature, 'o', color='w', markerfacecolor='orange')
    skew.shade_cape(full_pprof, full_Tprof, prof,color='orange')

    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 50)
    
    # Add the relevant special lines
    skew.plot_dry_adiabats(alpha=alpha_refprof)
    skew.plot_moist_adiabats(alpha=alpha_refprof)
    skew.plot_mixing_lines(alpha=alpha_refprof,pressure = np.linspace(1000,500,21) * units.hPa)
    
    return skew