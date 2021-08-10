import numpy as np

def Barnes2003_I(BV,t):
    '''
    interface gyrochrone
    eqns 1 & 2 from Barnes (2003)
    
    t in Myr
    '''
    P = np.sqrt(t) * np.sqrt(BV - 0.5) - 0.15 * (BV - 0.5)
    return P


def Barnes2003_C(BV,t):
    '''
    convective gyrochrone
    eqn 15 from Barnes (2003)
    
    t in Myr
    '''
    PI = Barnes2003_I(BV,t)
    P = 0.2 * np.exp(t / (100* (BV + 0.1 - (t/3000))**3))
    
    bd = np.where((P >= PI))[0]
    if len(bd) > 0:
        P[bd] = np.nan
    return P


def bv2teff(BV, logg=4.3, feh=0):
    """
    Relation from Sekiguchi & Fukugita (2000)
    https://arxiv.org/abs/astro-ph/9904299
    """
    # Full Sample, Tbl 3
    c = np.array([3.939654, -0.395361, 0.2082113, -0.0604097])
    f1, f2, g1, h1 = 0.027153, 0.005036, 0.007367, -0.01069
    
    logTeff = c[0] + c[1]*BV + c[2]*(BV**2) + c[3]*(BV**3) + \
               f1*feh + f2*(feh**2) + \
               g1*logg + h1*BV*logg
    return 10**logTeff

    
def BarnesPdot(BV, time):
    '''
    taking the derivative
    '''
    dt = time[1]-time[0] # in Myr
    P = Barnes2003_C(BV, time) # in days
    return np.gradient(P)/365.25/1e6 / dt # unitless



def OmC(BV,time, T=10, f=1):
    '''
    assume 10yr observing baseline
    
    following helpful math from JJ Hermes' webpage
    http://jjherm.es/research/omc.html
    '''
    P = Barnes2003_C(BV, time)
    Pdot = BarnesPdot(BV,time) * f # a fudge-factor to explore
    
    OC = Pdot / (2*P) * ((T*365.25)**2) # in days
    OC = OC * 24*60 # in min
    return OC

