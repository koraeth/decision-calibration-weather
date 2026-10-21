"""Wind conversions of the wind power task: 10 m wind speed -> hub-height wind speed -> normalised power.

The paper uses powerlaw_windspeed(v10, HUBHEIGHT) and then windspeed2power (numpy arrays).
"""
import numpy as np


HUBHEIGHT = 120  # in meters


def windspeed2power(y=None, v_cut_off=23):
    '''
    Convert wind speed (m/s) to normalized power [0,1] using a generic power curve.
    Assumes cut-in speed of 3 m/s, rated speed of 13 m/s, and cut-off speed of 23 m/s.
    '''
    power = np.zeros_like(y, dtype=float)
    v_cut_in=3
    v_rated=13

    mask1 = y < v_cut_in
    mask2 = (v_cut_in <= y) & (y < v_rated)
    mask3 = (v_rated <= y) & (y < v_cut_off)
    mask4 = y >= v_cut_off

    power[mask1] = 0
    power[mask2] = (y[mask2]**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3)
    power[mask3] = 1
    power[mask4] = 0

    return power


def powerlaw_windspeed(v10=None, hubheight=100, alpha=0.1):
    '''
    Convert wind speed at 10m to wind speed at hubheight using power law.
    Default alpha=0.1 is for open sea.
    '''
    return v10 * (hubheight / 10) ** alpha
