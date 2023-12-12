import sympy
theta = sympy.Symbol('theta')
psi = sympy.Symbol('psi')
phi = sympy.Symbol('phi')
def rotation_z(a):
    r = sympy.zeros(3, 3)
    r[2,2]=1
    r[0,0] = sympy.cos(a)
    r[1,1] = sympy.cos(a)
    r[0,1] = -sympy.sin(a)
    r[1, 0] = sympy.sin(a)
    return r
def rotation_y(a):
    r = sympy.zeros(3, 3)
    r[1,1]=1
    r[0,0] = sympy.cos(a)
    r[2,2] = sympy.cos(a)
    r[0,2] = sympy.sin(a)
    r[2, 0] = -sympy.sin(a)
    return r
def rotation_x(a):
    r = sympy.zeros(3, 3)
    r[0,0]=1
    r[1,1] = sympy.cos(a)
    r[2,2] = sympy.cos(a)
    r[1,2] = -sympy.sin(a)
    r[2, 1] = sympy.sin(a)
    return r
Rz = rotation_z(psi)
Ry = rotation_y(theta)
Rx = rotation_x(phi)

R = np.dot(Rz, np.dot(Ry, Rx))
R = sympy.simplify(R)
print(sympy.ccode(R))

theta = sympy.pi/2
Ry = rotation_y(theta)
R = np.dot(Rz, np.dot(Ry, Rx))
R = sympy.simplify(R)
print(sympy.ccode(R))

theta = -sympy.pi/2
Ry = rotation_y(theta)
R = np.dot(Rz, np.dot(Ry, Rx))
R = sympy.simplify(R)
print(sympy.ccode(R))

# ECEF to NED
def ECEF2NED():
    el = sympy.Symbol("el")
    az = sympy.Symbol("az")
    R = np.dot(rotation_y(el+sympy.pi/2), rotation_z(-az))
    R = sympy.simplify(R)
    return R

# ECEF to ENU
def ECEF2ENU():
    el = sympy.Symbol("el")
    az = sympy.Symbol("az")
    R = np.dot(rotation_z(-sympy.pi/2), np.dot(rotation_y(el-sympy.pi/2), rotation_z(-az)))
    R = sympy.simplify(R)
    return R