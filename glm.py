# -*- coding: utf-8 -*-
"""
NOTE: the final transformation matrix must be transposed and copied is ti will be used in OpenGL
"""

# Note: we use functions from math module because they're faster on scalars

import math
import numpy as np
# register gpuarray class in opengl
from OpenGL.plugins import FormatHandler
FormatHandler('glm','OpenGL.arrays.numpymodule.NumpyHandler',
        ['glm.Matrix4f', 'glm.Vector2', 'glm.Vector3'])


def concatenate(*matrices):
    M = np.identity(4, np.float32)
    for i in matrices:
        M = np.dot(M, i)
    return M

#-----------------------------------------------------------------------
def clampVector(v3, v1, v2):
    # Along x
    vmax = max(v1.x, v2.x)
    vmin = min(v1.x, v2.x)
    x = max(vmin, min(vmax, v3.x))

    # Along y
    vmax = max(v1.y, v2.y)
    vmin = min(v1.y, v2.y)
    y = max(vmin, min(vmax, v3.y))
    return vec2(x,y)


#-----------------------------------------------------------------------
def lengthVector(vec):
    return np.sqrt(np.sum(vec**2,axis=-1))


#-----------------------------------------------------------------------
def normalizeVector(vec):
    den = np.sqrt(np.sum(vec**2,axis=-1))
    if den == 0:
        return vec2(0,0)
    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1))).T


#------------------------------------------------------------------------------
def translationMatrix(x, y=None, z=None):
    """Translate by an offset (x, y, z) .

    Parameters
    ----------
    x : float
        X coordinate of a translation vector.
    y : float | None
        Y coordinate of translation vector. If None, `x` will be used.
    z : float | None
        Z coordinate of translation vector. If None, `x` will be used.

    Returns
    -------
    M : array
        Translation matrix
    """

    T = np.array([[1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return T


#------------------------------------------------------------------------------
def scaleMatrix(x, y=None, z=None):
    """Non-uniform scaling along the x, y, and z axes

    Parameters
    ----------
    M : array
        Original transformation (4x4).
    x : float
        X coordinate of the translation vector.
    y : float | None
        Y coordinate of the translation vector. If None, `x` will be used.
    z : float | None
        Z coordinate of the translation vector. If None, `x` will be used.

    Returns
    -------
    M : array
        Updated transformation (4x4). Note that this function operates
        in-place.
    """
    y = x if y is None else y
    z = x if z is None else z
    S = np.array([[x, 0.0, 0.0, 0.0],
                  [0.0, y, 0.0, 0.0],
                  [0.0, 0.0, z, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return S


#------------------------------------------------------------------------------
def rotationMatrix(x, y, z):
    rx = xrotate(x)
    ry = yrotate(y)
    rz = zrotate(z)

    return np.dot(rz, np.dot(ry, rx)).astype(np.float32)

#------------------------------------------------------------------------------
def xrotate(theta):
    """Rotate about the X axis
    """
    t = math.pi * theta / 180.
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, cosT, -sinT, 0.0],
                  [0.0, sinT, cosT, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


#------------------------------------------------------------------------------
def yrotate(theta):
    """Rotate about the Y axis
    """
    t = math.pi * theta / 180.
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = np.array(
        [[cosT, 0.0, sinT, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-sinT, 0.0, cosT, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


#------------------------------------------------------------------------------
def zrotate(theta):
    """Rotate about the Z axis
    """
    t = math.pi * theta / 180.
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = np.array(
        [[cosT, -sinT, 0.0, 0.0],
         [sinT, cosT, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


#------------------------------------------------------------------------------
def rotate(M, angle, x, y, z, point=None):
    """Rotation about a vector

    Parameters
    ----------
    M : array
        Original transformation (4x4).
    angle : float
        Specifies the angle of rotation, in degrees.
    x : float
        X coordinate of the angle of rotation vector.
    y : float | None
        Y coordinate of the angle of rotation vector.
    z : float | None
        Z coordinate of the angle of rotation vector.

    Returns
    -------
    M : array
        Updated transformation (4x4). Note that this function operates
        in-place.
    """
    angle = math.pi * angle / 180
    c, s = math.cos(angle), math.sin(angle)
    n = math.sqrt(x * x + y * y + z * z)
    x /= n
    y /= n
    z /= n
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    R = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, 0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0],
                  [0, 0, 0, 1]], dtype=M.dtype).T
    # Changed to match glm
    M[...] = np.dot(R, M)
    return M
#------------------------------------------------------------------------------
def ortho(left, right, bottom, top, znear, zfar):
    """Create orthographic projection matrix

    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.

    Returns
    -------
    M : array
        Orthographic projection matrix (4x4).
    """
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M


def projectionMatrix(fov, width, height, zNear, zFar):

    ar = width/float(height)
    tanHalfFov = math.tan(math.radians(fov/2.))
    zRange = zNear - zFar

    M = np.eye(4, dtype=np.float32)

    M[0,0] = 1./(ar*tanHalfFov)
    M[1,1] = 1./tanHalfFov
    M[2,2] = (-zNear-zFar)/(zNear-zFar)
    M[2,3] = (2*zFar*zNear)/(zNear-zFar)
    M[3,2] = 1.0

    return M

#------------------------------------------------------------------------------
def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum

    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.

    Returns
    -------
    M : array
        View frustum matrix (4x4).
    """
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M

def perspective(fovy, w, h, znear, zfar):
    """Create perspective projection matrix

    Parameters
    ----------
    fovy : float
        The field of view along the y axis.
    aspect : float
        Aspect ratio of the view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.

    Returns
    -------
    M : array
        Perspective projection matrix (4x4).
    """
    #aspet = width/float(height)
    #assert(znear != zfar)
    #h = math.tan(fovy / 360.0 * math.pi) * znear
    #w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def cameraMatrix(forward, up):
    f = forward.normalized()
    r = up.normalized()
    r = r.cross(f)
    u = f.cross(r)
    M = np.zeros((4,4), np.float32)

    M[0,0] = r.x
    M[0,1] = r.y
    M[0,2] = r.z
    M[1,0] = u.x
    M[1,1] = u.y
    M[1,2] = u.z
    M[2,0] = f.x
    M[2,1] = f.y
    M[2,2] = f.z
    M[3,3] = 1.0
    return M.T


#==============================================================================
"""Inspired from Pyrr """

class NpProxy(object):
    def __init__(self, index):
        self._index = index

    def __get__(self, obj, cls):
        return obj[self._index]

    def __set__(self, obj, value):
        obj[self._index] = value


#-----------------------------------------------------------------------
class Vector2(np.ndarray):
    _shape = (2,)

    x = NpProxy(0)
    y = NpProxy(1)
    xy = NpProxy([0,1])

    def __new__(cls, *args):

        l = len(args)
        if l ==1  or l > 2:
            raise ValueError('Wrong number of argumets!')
        elif l == 2:
            obj = np.array(args, dtype=None)
        else:
            obj = np.zeros(cls._shape, dtype=None)
        obj = obj.view(cls)
        return obj

    def __iadd__(self, other):
        self[:] = self.__add__(other)
        return self

    def __isub__(self, other):
        self[:] = self.__sub__(other)
        return self

    def __imul__(self, other):
        self[:] = self.__mul__(other)
        return self

    def __idiv__(self, other):
        self[:] = self.__div__(other)
        return self

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def max(self):
        return max(self.x, self.y)

    def dot(self, v):
        return self.x * v.x + self.y * v.y

    def normalized(self):
        length = self.length()
        return Vector2(self.x/length, self.y/length)

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length

    def cross(self, v):
        return self.x * v.y - self.y * v.x

    def rotate(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector2(self.x*cos-self.y*sin, self.x*sin+self.y*cos)

    def mul(self, v):
        return Vector2(self.x * v.x, self.y * v.y)

#------------------------------------------------------------------------------
class Vector3(np.ndarray):
    _shape = (3,)

    x = NpProxy(0)
    y = NpProxy(1)
    z = NpProxy(2)
    xyz = NpProxy([0,1,2])
    xz = NpProxy([0,2])
    yz = NpProxy([1,2])

    def __new__(cls, *args):

        l = len(args)
        if l == 1 and isinstance(args[0], np.ndarray):
            obj = args[0]
        elif l in [1,2]  or l > 3:
            raise ValueError('Wrong number of argumets!')
        elif l == 3:
            obj = np.array(args, dtype=None)
        else:
            obj = np.zeros(cls._shape, dtype=None)
        obj = obj.view(cls)
        return obj

    def __iadd__(self, other):
        self[:] = self.__add__(other)
        return self

    def __isub__(self, other):
        self[:] = self.__sub__(other)
        return self

    def __imul__(self, other):
        self[:] = self.__mul__(other)
        return self

    def __idiv__(self, other):
        self[:] = self.__div__(other)
        return self

    def length(self):
        return np.linalg.norm(self)

    def max(self):
        return self.max()

    def dot(self, v):
        return np.dot(self, v)

    def normalized(self):
        length = self.length()
        return Vector3(self.x/length, self.y/length, self.z/length)

    def normalize(self):
        length = self.length()
        self /= length

    def cross(self, v):
        return Vector3(np.cross(self, v))

    def rotate(self, angle, axis):
        rad = math.radians(angle/2.0)
        cosAngle = math.cos(rad)
        sinAngle = math.sin(rad)

        rX = axis.x * sinAngle
        rY = axis.y * sinAngle
        rZ = axis.z * sinAngle
        rW = cosAngle

        rotation = Quaternion(rX, rY, rZ, rW)
        conjugate = rotation.conjugate()
        w = rotation.mul(self).mul(conjugate)

        self.x = w.x
        self.y = w.y
        self.z = w.z

    def mul(self, v):
        return Vector3(self*v)


#-----------------------------------------------------------------------
class Quaternion(np.ndarray):
    _shape = (4,)

    x = NpProxy(0)
    y = NpProxy(1)
    z = NpProxy(2)
    w = NpProxy(3)
    xyzw = NpProxy([0,1,2,3])
    xyz = NpProxy([0,1,2])
    xyw = NpProxy([0,1,3])
    xzw = NpProxy([0,2,3])
    yzw = NpProxy([1,2,3])
    xz = NpProxy([0,2])
    xy = NpProxy([0,2])
    xw = NpProxy([0,3])
    yz = NpProxy([1,2])
    yw = NpProxy([1,3])
    zw = NpProxy([2,3])

    def __new__(cls, *args):

        l = len(args)
        if l in [1,2,3]  or l > 4:
            raise ValueError('Wrong number of argumets!')
        elif l == 4:
            obj = np.array(args, dtype=None)
        else:
            obj = np.zeros(cls._shape, dtype=None)
        obj = obj.view(cls)
        return obj

    def __iadd__(self, other):
        self[:] = self.__add__(other)
        return self

    def __isub__(self, other):
        self[:] = self.__sub__(other)
        return self

    def __imul__(self, other):
        self[:] = self.__mul__(other)
        return self

    def __idiv__(self, other):
        self[:] = self.__div__(other)
        return self

    def length(self):
        return math.sqrt(x * x + y * y + z * z + w * w)

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length
        self.w /= length
        return self

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def mul(self, q1):
        if isinstance(q1, Quaternion):
            ww = self.w * q1.w - self.x * q1.x - self.y * q1.y - self.z * q1.z
            xx = self.x * q1.w + self.w * q1.x + self.y * q1.z - self.z * q1.y
            yy = self.y * q1.w + self.w * q1.y + self.z * q1.x - self.x * q1.z
            zz = self.z * q1.w + self.w * q1.z + self.x * q1.y - self.y * q1.x
        elif isinstance(q1, Vector3):
            ww = -self.x * q1.x - self.y * q1.y -self.z * q1.z
            xx =  self.w * q1.x + self.y * q1.z -self.z * q1.y
            yy =  self.w * q1.y + self.z * q1.x -self.x * q1.z
            zz =  self.w * q1.z + self.x * q1.y -self.y * q1.x
        return Quaternion(xx, yy, zz, ww)


class Matrix4f(np.ndarray):
    _shape = (4,4)

    def __new__(cls, array=None):

        if array is None:
            array = np.identity(4, dtype=np.float32)

        obj = array.view(cls)
        return obj

    @staticmethod
    def translationMatrix(x, y, z):
        """ Create translation matrix from vector

        """
        m = np.identity(4, np.float32)
        m[0 , 3] = x
        m[1,  3] = y
        m[2 , 3] = z
        return Matrix4f(m)

    @staticmethod
    def rotationMatrix(x, y, z):
        """ Create rotation matrix from vector
            NOTE:
        """
        rx = Matrix4f()
        ry = Matrix4f()
        rz = Matrix4f()

        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

        rz[0,0] = math.cos(z)
        rz[0,1] = -math.sin(z)
        rz[1,0] = math.sin(z)
        rz[1,1] = math.cos(z)

        ry[0,0] = math.cos(y)
        ry[0,2] = -math.sin(y)
        ry[2,0] = math.sin(y)
        ry[2,2] = math.cos(y)

        rx[1,1] = math.cos(x)
        rx[2,1] = math.sin(x)
        rx[1,2] = -math.sin(x)
        rx[2,2] = math.cos(x)

        return Matrix4f(concatenate(rz, ry, rx))

    @staticmethod
    def scaleMatrix(x, y, z):
        """ Create scale matrix """
        m = np.identity(4, np.float32)

        m[0,0] = x
        m[1,1] = y
        m[2,2] = z

        return Matrix4f(m)

    @staticmethod
    def projectionMatrix(projInfo):

        ar = projInfo['width']/float(projInfo['height'])
        tanHalfFov = math.tan(math.radians(projInfo['fov']/2.))
        zRange = projInfo['zNear'] - projInfo['zFar']
        m = np.zeros((4,4), np.float32)

        m[0,0] = 1.0/(tanHalfFov*ar)
        m[1,1] = 1.0/tanHalfFov
        m[2,2] = (-projInfo['zNear']-projInfo['zFar'])/zRange
        m[3,2] = 1.0
        m[2,3] = 2. * projInfo['zFar'] * projInfo['zNear'] / zRange

        return Matrix4f(m)

    @staticmethod
    def initCamera(target, up):
        N = target.copy()
        N.normalize()

        U = up.copy()


        U = U.cross(N)
        U.normalize

        V = N.cross(U)

        m = np.identity(4, np.float32)

        m[0,0] = U.x;	m[0,1] = U.y;	m[0,2] = U.z;	m[0,3] = 0;
        m[1,0] = V.x;	m[1,1] = V.y;	m[1,2] = V.z;	m[1,3] = 0;
        m[2,0] = N.x;	m[2,1] = N.y;	m[2,2] = N.z;	m[2,3] = 0;
        m[3,0] = 0; m[3,1] = 0; m[3,2] = 0; m[3,3] = 1;

        return Matrix4f(m)

    def mul(self, r):
        Warning('Deprecated use array.dot!!!')
        return np.dot(self, r)

