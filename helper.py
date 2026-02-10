import numpy as np

def stumpS(z):
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z))**3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z,dtype=np.complex128)) - (np.sqrt(-z,dtype=np.complex128))) / (np.sqrt(-z,dtype=np.complex128))**3
    else:
        s = 1/6
    return s

def stumpC(z):
    if z > 0:
        s = (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        s = (np.cosh(np.sqrt(-z,dtype=np.complex128)) - 1) / (-z)
    else:
        s = 1/2
    return s

def y(z,r1,r2,A): # 5.38
    return r1 + r2 + A*(z*stumpS(z) - 1)/np.sqrt(stumpC(z))

def F(z,t, r1,r2, mu,A): # 5.40
    return ((y(z,r1,r2,A)/stumpC(z))**1.5) * stumpS(z) + A*np.sqrt(y(z,r1,r2,A)) - np.sqrt(mu)*t

def dFdz(z,t,r1,r2,mu,A): # 5.43
    if z == 0:
        result = (np.sqrt(2)/40) * y(0,r1,r2,A)**(1.5) + (A/8)*(np.sqrt(y(0,r1,r2,A)) + A*(np.sqrt(1/(2*y(0,r1,r2,A))))) 
    
    else:
        first_part = (y(z,r1,r2,A)/stumpC(z))**1.5
        second_part = ((1/(2*z)) * (stumpC(z) - (1.5 * (stumpS(z) / stumpC(z)))) + (0.75 * (stumpS(z)**2  / stumpC(z))) )
        third_part = (A/8) * (3* (stumpS(z)/stumpC(z)) * np.sqrt(y(z,r1,r2,A)) + A*(np.sqrt(stumpC(z)/y(z,r1,r2,A))))
        result = first_part*second_part + third_part
    
    return result


# def lambert_solve(R1,R2,t, trajectory_type, debug = False):
#     mu = 3.986 * 10**5  # Earths gravitational parameter in (km^3/s^2) 

#     r1 = np.linalg.norm(R1) # Mag of R1
#     r2 = np.linalg.norm(R2) # Mag of R2

#     cross_r1_r2 = np.cross(R1,R2)
#     theta = np.arccos(np.dot(R1,R2)/(r1*r2))

#     ## Include thing about prograde/retrograde

#     A = np.sin(theta) * np.sqrt((r1*r2) / (1-np.cos(theta)))

#     ## Determine where F(z,t) changes sign. Use value of z as starting value  for 5.45
#     z = -100
#     while F(z,t,r1,r2,mu,A) < 0:
#         z = z + 0.1

#     ## Error Tolerance and max number of iterations
#     tol = 1 * 10**(-8)
#     nmax = 5000

#     # Iterate on 5.45 until z is within error tolerance. Just Newton_Raphson iteration
#     ratio = 1
#     n = 0
#     while (np.abs(ratio) > tol) and (n<=nmax):
#         n = n + 1
#         ratio = F(z,t,r1,r2,mu,A) / dFdz(z,t,r1,r2,mu,A)
#         z = z - ratio

#     if n >= nmax:
#         print(f"\n Numeber of iterations exceeds {nmax}")

#     # Lagrange Coefficients
#     f = 1 - (y(z,r1,r2,A)/r1) # 5.46a
#     g = A*np.sqrt(y(z,r1,r2,A)/mu) # 5.46b
    
#     fdot = (np.sqrt(mu)/(r1*r2)) * (np.sqrt(y(z,r1,r2,A)/stumpC(z)) * (z*stumpS(z) -1)) # 5.46c
#     gdot = 1 - (y(z,r1,r2,A)/r2) # 5.46d


  

#     V1 = (1/g)*(R2 - f*R1)  #5.28
#     V2 = (1/g)*(gdot*R2 - R1)   #5.29

#     if debug:
#         print(f"{r1 = }\t{r2 = }\t{cross_r1_r2 = }\t{A = } \n {V1 = }\t{V2 = }")

#     return V1, V2


class Orbit:
    def __init__(self,h,i,RA,e,w,theta):
        self.h = h
        self.i = i
        self.RA = RA
        self.e = e
        self.w = w
        self.theta = theta


def calculate_orbit_from_state_vector(r,v,mu,debug = True):
    mag_r = np.linalg.norm(r)
    mag_v = np.linalg.norm(v)

    radial_v = np.dot(r,v/mag_r)

    h = np.cross(r,v)
    mag_h = np.linalg.norm(h)

    i = np.degrees(np.arccos(h[2]/mag_h))

    node_line = np.cross([0,0,1],h)
    
    mag_node_line = np.linalg.norm(node_line)

    if node_line[1] >= 0:
        RA = np.degrees(np.arccos(node_line[0]/mag_node_line))

    else:
        RA = 360 - np.degrees(np.arccos(node_line[0]/mag_node_line))

    e_vector = (1/mu) * (np.cross(v,h) - (mu*(r/mag_r)))
    e = np.linalg.norm(e_vector)

    if e_vector[2] >= 0:
        w = np.degrees(np.arccos((np.dot(node_line,e_vector)/(mag_node_line*e))))

    else:
        w = 360 - np.degrees(np.arccos((np.dot(node_line,e_vector)/(mag_node_line*e))))

    if radial_v >= 0:
        theta = np.degrees(np.arccos((np.dot(e_vector,r)/(e*mag_r))))

    else:
        theta = 360 - np.degrees(np.arccos((np.dot(e_vector,r)/(e*mag_r))))
    
    if debug:
        print(f"""
        {r = }
        {mag_r = }
        {v = }
        {mag_v = }
        {radial_v = }
        {h = }
        {i = }
        {RA = }
        {e = }
        {w = }
        {theta = }""")

    orbit = Orbit(mag_h,i,RA,e,w,theta)
    return orbit

def calculate_state_vector_from_orbit(orbit:Orbit , mu, debug = False):
    r_perifocal = ((orbit.h**2/mu) * (1/(1+orbit.e*np.cos(np.radians(orbit.theta))))) * np.array([np.cos(np.radians(orbit.theta)), np.sin(np.radians(orbit.theta)), 0])

    v_perifocal = (mu/orbit.h) * np.array([-np.sin(np.radians(orbit.theta)), orbit.e+np.cos(np.radians(orbit.theta)),0])

    R3_RA = np.array([[np.cos(np.radians(orbit.RA)), np.sin(np.radians(orbit.RA)), 0] ,
                      [-np.sin(np.radians(orbit.RA)), np.cos(np.radians(orbit.RA)), 0],
                        [0,0,1]  ])
    
    R3_w = np.array([[np.cos(np.radians(orbit.w)), np.sin(np.radians(orbit.w)), 0] ,
                      [-np.sin(np.radians(orbit.w)), np.cos(np.radians(orbit.w)), 0],
                        [0,0,1]  ])

    R1_i = np.array([[1,0,0],
                     [0, np.cos(np.radians(orbit.i)), np.sin(np.radians(orbit.i))],
                     [0, -np.sin(np.radians(orbit.i)), np.cos(np.radians(orbit.i))]])
    
    QX_p = R3_w @ R1_i @  R3_RA  # @ is matrix multiply, can also be done with .dot(). QX_p is transform matrix from geocentric to perifocal frame
    
    # Qp_X - transform matrix from perifocal to geocentric is QX_p.T (ie. transpose)

    r_geo = QX_p.T @ r_perifocal
    v_geo = QX_p.T @ v_perifocal
    if debug:
        print(f"{r_perifocal = }\n{v_perifocal = }\n{QX_p = }\n{r_geo = }\n{v_geo = }")

    return r_geo, v_geo


def orbit_eqn(h,e,theta,mu,debug = True):
    r = ((h**2)/mu) * (1/(1 + (e*np.cos(np.radians(theta)))))
    return r

def semimajor_axis(perigee,apogee):
    return (1/2)*(perigee + apogee)

def period(a,mu): # a is semimajor axis 
    return (( 2*np.pi/ ((mu)**(1/2))) * a**(3/2) )/(60*60)  # Period in hours