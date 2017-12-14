from glm import Vector3, Matrix4f
import numpy as np


#self._m_scale = glm.Vector3(1., 1., 1.)
#self._m_rotation = glm.Vector3(0., 0., 0.)
#self._m_pos = glm.Vector3(0., 0., 0.)

def concatenate(*matrices):
    M = np.identity(4, np.float32)
    for i in matrices:
        M = np.dot(M, i)
    return M

class Pipeline(object):
    def __init__(self):

        self.m_scale      = Vector3(1.0, 1.0, 1.0)
        self.m_worldPos   = Vector3(0.0, 0.0, 0.0)
        self.m_rotateInfo = Vector3(0.0, 0.0, 0.0) 
        self.m_persProjInfo = None
        self.m_orthoProjInfo = None
        self._camera = None
    
    def scale(self, sx, sy, sz):
        self.m_scale = Vector3(sx, sy, sz)
        
    def worldPos(self, x, y, z):
        self.m_worldPos   = Vector3(x, y, z)
        
    def rotate(self, rx, ry, rz):
        self.m_rotateInfo = Vector3(rx, ry, rz)
        
    def setPerspectiveProj(self, fov, width, height, zNear, zFar):
        self.m_persProjInfo =     {'fov'   : fov,
                                   'height': height,
                                   'width' : width,
                                   'zNear' : zNear,
                                   'zFar'  :  zFar}
        
    def setOrthographicProj(self, proj):
        self.m_orthoProjInfo = proj     
        
    def setCamera(self, camera):
        self._camera = camera


    def getWorldTrans(self):
        ScaleTrans = Matrix4f.scaleMatrix(self.m_scale.x, self.m_scale.y, self.m_scale.z)
        RotateTrans = Matrix4f.rotationMatrix(self.m_rotateInfo.x, self.m_rotateInfo.y, self.m_rotateInfo.z)
        TranslationTrans = Matrix4f.translationMatrix(self.m_worldPos.x, self.m_worldPos.y, self.m_worldPos.z)
        matrix = concatenate(TranslationTrans, RotateTrans, ScaleTrans)
        return Matrix4f(matrix)

    def getWorldProjdTrans(self):
        transformation = self.getWorldTrans()
        projection = Matrix4f.projectionMatrix(self.m_persProjInfo)
        matrix = np.dot(projection, transformation)
        return matrix

    def getViewTransform(self):
        cameraTranslation = Matrix4f.translationMatrix(-self._camera._pos.x, -self._camera._pos.y, -self._camera._pos.z)
        cameraRotation = Matrix4f.initCamera(self._camera._target, self._camera._up)
        matrix = np.dot(cameraTranslation, cameraRotation)
        return Matrix4f(matrix)

    def getViewProjTransform(self):
        projection = Matrix4f.projectionMatrix(self.m_persProjInfo)  # projection
        view = self.getViewTransform() # view
        matrix = np.dot(projection, view)
        return Matrix4f(matrix)

    def getWorldViewProjdTrans(self):
        transformation = self.getWorldTrans() # world
        viewProjection = self.getViewProjTransform() # view projection

        matrix = np.dot(viewProjection, transformation)
        return Matrix4f(matrix)

    def getProjectionMatrix(self):
        return Matrix4f.projectionMatrix(self.m_persProjInfo)  # projection
