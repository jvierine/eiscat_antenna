#!/usr/bin/env python
#
# (c) 2016-2018 Juha Vierinen
#
# Reference:
# https://www.cv.nrao.edu/course/astr534/2DApertures.html
#
# - Uniformly filled circular aperture of radius a
# - Cassegrain antenna with radius a0 and subreflector radius a1
# - Planar gaussian illuminated aperture (approximates a phased array)
#
import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import scipy.special as s
import scipy.interpolate as sint

import jcoord


# a = radius
# f = frequency
# I_0 = gain at center
def airy(theta,a=40.0,f=233e6,I_0=10**4.3):
    lam=c.c/f
    k=2.0*n.pi/lam
    return(I_0*((2.0*s.jn(1,k*a*n.sin(theta))/(k*a*n.sin(theta))))**2.0)

uhf_data=n.array([[-1.0000000e+03, -5.0000000e+01],
                [-1.0000000e+01, -5.0000000e+01],
                [-4.9832220e+00, -4.1696290e+01],
                [-4.9194630e+00, -4.1380800e+01],
                [-4.8355710e+00, -4.0960146e+01],
                [-4.7550340e+00, -4.0539492e+01],
                [-4.6677860e+00, -3.9961093e+01],
                [-4.5704710e+00, -3.9172367e+01],
                [-4.4832220e+00, -3.7542332e+01],
                [-4.3926190e+00, -3.6648443e+01],
                [-4.2684580e+00, -3.5596808e+01],
                [-4.3456390e+00, -3.6017462e+01],
                [-4.1979880e+00, -3.5439063e+01],
                [-4.1308740e+00, -3.5859716e+01],
                [-4.0738270e+00, -3.6332952e+01],
                [-4.0335590e+00, -3.6543279e+01],
                [-3.9865790e+00, -3.6595861e+01],
                [-3.9328880e+00, -3.6332952e+01],
                [-3.8993310e+00, -3.5544226e+01],
                [-3.8590620e+00, -3.4650336e+01],
                [-3.8154380e+00, -3.3703865e+01],
                [-3.7617470e+00, -3.2547066e+01],
                [-3.7114120e+00, -3.1390268e+01],
                [-3.6610760e+00, -3.0233470e+01],
                [-3.6107410e+00, -2.9549907e+01],
                [-3.5402710e+00, -2.9129253e+01],
                [-3.4530230e+00, -2.9286998e+01],
                [-3.3926200e+00, -3.0286051e+01],
                [-3.3389290e+00, -3.1442850e+01],
                [-3.2885940e+00, -3.3020302e+01],
                [-3.2651040e+00, -3.4334846e+01],
                [-3.2315470e+00, -3.5964880e+01],
                [-3.2114130e+00, -3.7647496e+01],
                [-3.1912780e+00, -3.9277530e+01],
                [-3.1711440e+00, -4.1380800e+01],
                [-3.1476540e+00, -4.3273743e+01],
                [-3.1409430e+00, -4.4798614e+01],
                [-3.1275200e+00, -4.7059629e+01],
                [-3.1140970e+00, -4.9005153e+01],
                [-3.0838960e+00, -4.9583553e+01],
                [-3.0369160e+00, -4.9268062e+01],
                [-3.0167820e+00, -4.6849302e+01],
                [-3.0067150e+00, -4.4115051e+01],
                [-2.9731580e+00, -4.2379853e+01],
                [-3.0201380e+00, -4.5114104e+01],
                [-2.9530240e+00, -4.0960146e+01],
                [-2.9094000e+00, -3.9908511e+01],
                [-2.8523530e+00, -3.9698184e+01],
                [-2.8020170e+00, -4.0434329e+01],
                [-2.7617490e+00, -4.1012728e+01],
                [-2.7248360e+00, -4.0434329e+01],
                [-2.7047020e+00, -3.9382694e+01],
                [-2.6711450e+00, -3.7700078e+01],
                [-2.6208100e+00, -3.4334846e+01],
                [-2.5838970e+00, -3.1495431e+01],
                [-2.5436290e+00, -2.9339580e+01],
                [-2.4932930e+00, -2.7288892e+01],
                [-2.4362460e+00, -2.5343367e+01],
                [-2.3624210e+00, -2.3818496e+01],
                [-2.2919510e+00, -2.3134934e+01],
                [-2.2281930e+00, -2.2819443e+01],
                [-2.1543680e+00, -2.3555588e+01],
                [-2.0771860e+00, -2.5133040e+01],
                [-2.0369180e+00, -2.6447584e+01],
                [-1.9832270e+00, -2.9181835e+01],
                [-2.0033610e+00, -2.7604382e+01],
                [-1.9597370e+00, -3.1758340e+01],
                [-1.9060460e+00, -4.0171420e+01],
                [-1.8691330e+00, -4.5429595e+01],
                [-1.8523550e+00, -4.6323484e+01],
                [-1.8087310e+00, -4.2379853e+01],
                [-1.8020190e+00, -3.9172367e+01],
                [-1.7751740e+00, -3.6753606e+01],
                [-1.7516840e+00, -3.4019355e+01],
                [-1.7315500e+00, -3.2178994e+01],
                [-1.6979930e+00, -3.0443797e+01],
                [-1.6476570e+00, -2.9392162e+01],
                [-1.5771870e+00, -2.9286998e+01],
                [-1.5436300e+00, -3.0811869e+01],
                [-1.5134290e+00, -3.2231576e+01],
                [-1.4798720e+00, -3.3125466e+01],
                [-1.4496710e+00, -3.3598701e+01],
                [-1.4127580e+00, -3.3072884e+01],
                [-1.3758450e+00, -3.0969614e+01],
                [-1.3557110e+00, -2.7709546e+01],
                [-1.3120870e+00, -2.5238204e+01],
                [-1.2852420e+00, -2.3397842e+01],
                [-1.2349060e+00, -2.1136827e+01],
                [-1.1812150e+00, -1.8875812e+01],
                [-1.1375910e+00, -1.7613850e+01],
                [-1.0872550e+00, -1.6930287e+01],
                [-1.0402760e+00, -1.6351888e+01],
                [-9.8994000e-01, -1.6194143e+01],
                [-9.2618200e-01, -1.6719960e+01],
                [-9.1611500e-01, -1.7245778e+01],
                [-8.7920200e-01, -1.8612903e+01],
                [-8.4564500e-01, -2.0716173e+01],
                [-7.9866500e-01, -2.5395949e+01],
                [-8.2215500e-01, -2.2977188e+01],
                [-7.9195400e-01, -2.7446637e+01],
                [-7.8859800e-01, -2.9444743e+01],
                [-7.8859800e-01, -3.1022196e+01],
                [-7.6846400e-01, -3.2652230e+01],
                [-7.6175300e-01, -3.5491644e+01],
                [-7.4833000e-01, -3.6122625e+01],
                [-7.2484000e-01, -3.2441903e+01],
                [-7.0806100e-01, -2.7341473e+01],
                [-6.9128300e-01, -2.3503006e+01],
                [-6.7114900e-01, -1.9980029e+01],
                [-6.5101500e-01, -1.7456105e+01],
                [-6.0067900e-01, -1.3617637e+01],
                [-6.3423600e-01, -1.5510580e+01],
                [-5.7383300e-01, -1.1514367e+01],
                [-5.2685400e-01, -9.2007710e+00],
                [-5.0648000e-01, -8.5227890e+00],
                [-4.8130100e-01, -7.4419030e+00],
                [-4.5511500e-01, -6.3103500e+00],
                [-4.2792200e-01, -5.1450190e+00],
                [-3.8763500e-01, -4.0810220e+00],
                [-3.3828400e-01, -2.8988020e+00],
                [-2.7886200e-01, -1.9530270e+00],
                [-2.3051800e-01, -1.2436950e+00],
                [-1.8519600e-01, -8.5525100e-01],
                [-1.1167300e-01, -3.4858600e-01],
                [-6.2322000e-02, -1.7969700e-01],
                [ 1.0957000e-02,  0.0000000e+00],
                [ 0.0000000e+00,  0.0000000e+00]])


eiscat_uhf_fun_internal=sint.interp1d(n.pi*n.abs(uhf_data[:,0])/180.0,10**(uhf_data[:,1]/10.0))
def eiscat_uhf_fun(theta,peak_gain=10**4.8):
    """
    theta is in radians
    """
    return(peak_gain*eiscat_uhf_fun_internal(n.abs(theta)))



#
# A better model of the EISCAT UHF antenna
# 
def cassegrain(theta,a0=32.0,a1=4.58,f=930e6,I_0=10**4.8):
    """
    circular aperture with circular exclusion
    
    theta is angle from off-axis position in radians
    a0=antenna diameter
    a1=exclusion (subreflector) diameter
    f=frequency
    I_0=peak gain
    """
    lam=c.c/f
    k=2.0*n.pi/lam
    
    A=(I_0*((lam/(n.pi*n.sin(theta)))**2.0))/((a0**2.0-a1**2.0)**2.0)
    B=(a0*s.jn(1,a0*n.pi*n.sin(theta)/lam)-a1*s.jn(1,a1*n.pi*n.sin(theta)/lam))**2.0
    A0=(I_0*((lam/(n.pi*n.sin(1e-6)))**2.0))/((a0**2.0-a1**2.0)**2.0)
    B0=(a0*s.jn(1,a0*n.pi*n.sin(1e-6)/lam)-a1*s.jn(1,a1*n.pi*n.sin(1e-6)/lam))**2.0
    const=I_0/(A0*B0)
    return(A*B*const)

class beam_pattern:
    # az and el are the pointing direction
    # az0 and el0 are the on-axis direction
    def __init__(self,az0,el0,lat,lon,I_0=10**4.8,f=930e6,a0=16.0):
        self.on_axis=jcoord.azel_ecef(lat, lon, 0.0, az0, el0)
        self.lat=lat
        self.lon=lon
        self.I_0=I_0
        self.a0=a0
        self.f=f
        self.az0=az0
        self.el0=el0
        
    # move antenna towards direction az0 el0
    def point(self,az0,el0):
        self.az0=az0
        self.el0=el0        
        self.on_axis=jcoord.azel_ecef(self.lat, self.lon, 0.0, az0, el0)

    # directly apply on-axis position
    def point_k0(self,k0):
        self.on_axis=k0

    # get angle to on axis direction
    def angle(self,az,el):
        direction=jcoord.azel_ecef(self.lat, self.lon, 0.0, az, el)
        return(angle_deg(self.on_axis,direction))
    
    # gain for pointing direction k (ECEF)
    def gain(self,k):
        return(airy(n.pi*jcoord.angle_deg(self.on_axis,k)/180.0,a=self.a0,f=self.f,I_0=self.I_0))

class airy_beam(beam_pattern):
    # gain for pointing direction k
    def gain(self,k):
        return(airy(n.pi*jcoord.angle_deg(self.on_axis,k)/180.0,a=self.a0,f=self.f,I_0=self.I_0))

class cassegrain_beam(beam_pattern):
    def __init__(self,az0,el0,lat,lon,I_0=10**4.8,f=930e6,a0=16.0,a1=4.58):    
        beam_pattern.__init__(self,az0,el0,lat,lon,I_0=I_0,f=f,a0=a0)
        self.a1=a1
    # gain for pointing direction k
    def gain(self,k):
        return(cassegrain(n.pi*jcoord.angle_deg(self.on_axis,k)/180.0,a0=2.0*self.a0,a1=self.a1,f=self.f,I_0=self.I_0))
    
#
# Simple planar circular gaussian illuminated beam
# az0 and el0 determine the antenna phasing direction
# az1 and el1 determine the plane normal direction
#
class planar_beam(beam_pattern):
    def __init__(self,az0,el0,lat,lon,I_0=10**4.8,f=230e6,a0=32.0,az1=90,el1=90):    
        beam_pattern.__init__(self,az0,el0,lat,lon,I_0=10**4.8,f=930e6,a0=a0)
        self.plane_normal=jcoord.azel_ecef(lat, lon, 0.0, az1, el1) 
        self.lam=c.c/f
        self.I_0=I_0
        self.point(az0,el0)

    # directly apply on-axis position
    def point_k0(self,k0):
        self.on_axis=k0
        if n.abs(1-n.dot(self.on_axis,self.plane_normal)) < 1e-6:
            rd=n.random.randn(3)
            rd=rd/n.sqrt(n.dot(rd,rd))
            ct=n.cross(self.on_axis,rd)
        else:
            ct=n.cross(self.on_axis,self.plane_normal)
            
        ct=ct/n.sqrt(n.dot(ct,ct))
        ht=n.cross(self.plane_normal,ct)
        ht=ht/n.sqrt(n.dot(ht,ht))
        angle=jcoord.angle_deg(self.on_axis,ht)
  #      print(angle)
        ot=n.cross(self.on_axis,ct)
        ot=ot/n.sqrt(n.dot(ot,ot))

        self.I_1=n.sin(n.pi*angle/180.0)*self.I_0
        self.a0p=n.sin(n.pi*angle/180.0)*self.a0

        self.ct=ct
        self.ht=ht        
        self.ot=ot
        self.angle=angle

        self.sigma1=0.7*self.a0p/self.lam
        self.sigma2=0.7*self.a0/self.lam
        
    # point the antenna towards az0 and el0
    def point(self,az0,el0):
        k0=jcoord.azel_ecef(self.lat, self.lon, 0.0, az0, el0)
        self.az0=az0
        self.el0=el0
        self.point_k0(k0)
        
    # gain for pointing direction k
    def gain(self,k):
        # first, let's determine what the effective antenna looks like in the worst direction
        k0=k/n.sqrt(n.dot(k,k))
        
        A=n.dot(k0,self.on_axis)
        kda=A*self.on_axis
        l1=n.dot(k0,self.ct)
        kdc=l1*self.ct
        m1=n.dot(k0,self.ot)
        kdo=m1*self.ot
        
        l2=l1*l1
        m2=m1*m1
        gain1=self.I_1*n.exp(-n.pi*m2*2.0*n.pi*self.sigma1**2.0)*n.exp(-n.pi*l2*2.0*n.pi*self.sigma2**2.0)
        return(gain1)

def test_planar():
    for phased_el in n.linspace(3,90,num=10):
        print(phased_el)
        bp=planar_beam(0.0,phased_el,60,10,az1=0.0,el1=90.0,a0=40.0)
 #       print(bp.I_1)
        gains=[]
        els=n.linspace(0.0,90.0,num=1000)
        for ei,e in enumerate(els):
            k=jcoord.azel_ecef(60.0, 10.0, 0.0, 0.0, e)
#            print("phased el %f k el %f"%(phased_el,e))
            g=bp.gain(k)
            gains.append(g)
        gains=n.array(gains)
        plt.plot(els,10.0*n.log10(gains),label="el=%1.2f"%(phased_el))
        
        plt.ylim([0,50])
        plt.axvline(phased_el,color="black")
    plt.legend()
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Gain (dB)")
    plt.title("Planar array gain as a function of pointing direction")
    plt.show()

def test_planar4():
    bp=planar_beam(0.0,45.0,60,10,az1=0.0,el1=90.0,a0=40.0)    
    for phased_el in n.linspace(3,90,num=10):
        bp.point(0.0,phased_el)
        gains=[]
        els=n.linspace(0.0,90.0,num=1000)
        for ei,e in enumerate(els):
            k=jcoord.azel_ecef(60.0, 10.0, 0.0, 0.0, e)
            g=bp.gain(k)
            gains.append(g)
        gains=n.array(gains)
        plt.plot(els,10.0*n.log10(gains),label="el=%1.2f"%(phased_el))
        
        plt.ylim([0,50])
        plt.axvline(phased_el,color="black")
    plt.legend()
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Gain (dB)")
    plt.title("Planar array gain as a function of pointing direction")
    plt.show()
    
def test_planar2():
    el_phase=30.0
    az_phase=40.0    
    B=n.zeros([500,500])
    els=n.linspace(0,90,num=500)
    azs=n.linspace(0,360,num=500)
    bp=planar_beam(az_phase,el_phase,60,19,az1=0.0,el1=90.0)
    for ei,e in enumerate(els):
        for ai,a in enumerate(azs):
            k=jcoord.azel_ecef(60.0, 19.0, 0.0, a, e)
            B[ei,ai]=bp.gain(k)
    dB=10.0*n.log10(B)
    m=n.max(dB)
    plt.pcolormesh(azs,els,10.0*n.log10(B),vmin=m-20.0,vmax=m)
    plt.axhline(el_phase)
    plt.axvline(az_phase)    
    plt.colorbar()
    plt.show()
        
def test_planar3():
    S=n.zeros([100,200])
    el_phase=90.0
    bp=planar_beam(0,el_phase,60,19.0,az1=0.0,el1=el_phase)
    els=n.linspace(0,90,num=100)
    azs=n.linspace(0,360,num=200)
    
    for ei,e in enumerate(n.linspace(0,90,num=100)):
        for ai,a in enumerate(n.linspace(0,360,num=200)):
            k=jcoord.azel_ecef(60.0, 19.0, 0.0, a, e)
            S[ei,ai]=bp.gain(k)
    plt.pcolormesh(azs,els,10.0*n.log10(B),vmin=0,vmax=100)
    plt.axvline(el_phase)
    plt.colorbar()
    plt.show()


def plot_e3d():
    theta=n.linspace(-n.pi/2.0,n.pi/2.0,num=1000)
    beam_pattern=airy(theta)

    plt.plot(180.0*theta/n.pi,10.0*n.log10(beam_pattern))
    plt.xlabel("Angle off axis (deg)")
    plt.ylabel("Gain (dB)")    

    plt.savefig("airy.png")
    plt.show()

def plot_beams():
    bp=airy_beam(90.0,90,60,19,f=230e6,a0=16.0)
    gains=[]
    els=n.linspace(0,90.0,num=1000)
    for a in els:
        k=jcoord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)
    plt.plot(els,10.0*n.log10(gains),label="airy")

    bp=cassegrain_beam(90.0,90,60,19,f=230e6,a0=16.0)
    gains=[]
    for a in els:
        k=jcoord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)
    plt.plot(els,10.0*n.log10(gains),label="cassegrain")

    bp=planar_beam(0,90.0,60,19,I_0=10**4.8,f=230e6,a0=16.0,az1=0,el1=90.0)
    gains=[]
    for a in els:
        k=jcoord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)
    plt.plot(els,10.0*n.log10(gains),label="planar")
    plt.ylim([0,50])
    
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    theta=n.pi*n.linspace(-5,5,num=1000)/180.0
    plt.plot(180.0*theta/n.pi,10.0*n.log10(cassegrain(theta)))
    plt.plot(180.0*theta/n.pi,10.0*n.log10(eiscat_uhf_fun(theta)))
    plt.show()

    
    plt.show()
#    test_planar4()
 #   plot_beams()    
  #  test_planar()        
   # test_planar2()    

  
