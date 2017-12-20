from __future__ import division
import warnings
warnings.simplefilter('default')
import os
import numpy as np
import ctypes
import random
import glm
import OpenGL.GL as gl

# Local imports
from shader import Shader
from texture import Texture

# Fix ramdom sequence
np.random.seed(123)

GLSLDIR = 'glsl'

#------------------------------------------------------------------------------
def makeProgram(vertexfile, fragmentfile):
    return Shader(os.path.join(GLSLDIR, vertexfile),
            os.path.join(GLSLDIR, fragmentfile))

#------------------------------------------------------------------------------
def field2RGB(field):
    """ Return 2D field converted to uint8 RGB image (i.e. scaled in [0, 255])
    """
    rows = field.shape[0]
    cols = field.shape[1]
    u = field[:, :, 0]
    v = field[:, :, 1]
    uMin = u.min()
    uMax = u.max()
    vMin = v.min()
    vMax = v.max()
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255 * (u - uMin) / ( uMax - uMin)
    rgb[:, :, 1] = 255 * (v - vMin) / ( vMax - vMin)
    return rgb, uMin, uMax, vMin, vMax

#------------------------------------------------------------------------------
def modulus(field):
    """ Return normalized modulus of 2D field image (i.e. scaled in [0, 1.])
    """
    modulus = np.flipud(np.hypot(field[:, :, 0], field[:,:,1]))
    return np.asarray(modulus/modulus.max(), np.float32)

#==============================================================================
class FieldAnimation(object):
    """ Field Animation plot class
    """
    def __init__(self, width, height, field):
        """ Animate 2D vector field
        """
        # Parameters that can be changed later
        self.periodic = True
        self.drawField = False
        self.fadeOpacity = 0.996
        self.dropRateBump = 0.01
        self.speedFactor = 0.25
        self.dropRate = 0.003
        self.palette = True
        self.color = (0.5, 1.0, 1.0)
        self.pointSize = 1.0
        self._tracersCount = 16384
        self.fieldScaling = 0.0001

        # These are fixed
        self.w_width = width
        self.w_height = height

        # Prepare the data
        self._fieldAsRGB, uMin, uMax, vMin, vMax = field2RGB(field)
        # Compute field modulus
        self.modulus = modulus(field)

        # Since points are in [0, 1] a traslation and a scaling is needed on
        # the model matrix
        T = glm.Matrix4f.translationMatrix(-1., 1., 0)
        S = glm.Matrix4f.scaleMatrix(2., -2., 1)
        # Model transfgorm matrix
        model = np.dot(T,S)
        # View matrix
        view = glm.Matrix4f()
        # Projection matrix
        proj = glm.Matrix4f()
        self._MVP = np.dot(model, np.dot(view, proj))

        # Create a buffer for the tracers
        self.emptyPixels = np.zeros((width * height * 4), np.uint8)

        # CubeHelix parameters
        cubeHelixParams =(
                ('start', 'f'),
                ('gamma', 'f'),
                ('rot', 'f'),
                ('reverse', 'b'),
                ('minSat', 'f'),
                ('maxSat', 'f'),
                ('minLight', 'f'),
                ('maxLight', 'f'),
                ('startHue', 'f'),
                ('endHue', 'f'),
                ('useHue', 'b'),
                )

        # Create Shader program for the vector field
        self.fieldProgram = makeProgram('field.vert', 'field.frag')
        self.fieldProgram.addUniforms((('gMap', 'i'), ) + cubeHelixParams)

        # Create Shader program for the tracers
        self.drawProgram = makeProgram('draw.vert', 'draw.frag')
        self.drawProgram.addUniforms((
            ('MVP', '4fv'),
            ('u_tracers', 'i'),
            ('u_tracersRes', 'f'),
            ('palette', 'b'),
            ('pointSize', 'f'),
            ('u_field', 'i'),
            ('u_fieldMin', '2f'),
            ('u_fieldMax', '2f')) + cubeHelixParams)
        # Set values that will not change
        #TODO: mettere il bind in set uniforms!!
        self.drawProgram.bind()
        self.drawProgram.setUniform('u_fieldMin', (uMin, vMin))
        self.drawProgram.setUniform('u_fieldMax', (uMax, vMax))
        self.drawProgram.unbind()

        # Create Shader program for updating the screen
        self.screenProgram = makeProgram('quad.vert', 'screen.frag')
        self.screenProgram.addUniforms((
            ('u_screen', 'i'),
            ('u_opacity', 'f')))

        # Create Shader program for updating the tracers position
        self.updateProgram = makeProgram('quad.vert', 'update.frag')
        from IPython import embed; embed()
        self.updateProgram.addUniforms((
                ('u_tracers', 'i'),
                ('u_field', 'i'),
                ('u_fieldRes', '2f'),
                ('u_fieldMin', '2f'),
                ('u_fieldMax', '2f'),
                ('u_rand_seed', 'f'),
                ('u_speed_factor', 'f'),
                ('u_drop_rate', 'f'),
                ('u_drop_rate_bump', 'f'),
                ('fieldScaling', 'f'),
                ('periodic', 'b')))
        # Set values that will not change
        self.updateProgram.bind()
        self.updateProgram.setUniform('u_fieldMin', (uMin, vMin))
        self.updateProgram.setUniform('u_fieldMax', (uMax, vMax))
        self.updateProgram.unbind()

        self.initTracers()

    def setRenderingTarget(self, texture):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frameBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, texture.handle(), 0)

    def resetRenderingTarget(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.w_width, self.w_height)

    @property
    def tracersCount(self):
        return  self._tracersCount

    @tracersCount.setter
    def tracersCount(self, value):
        self._tracersCount = value
        self.initTracers()

    def initTracers(self):
        # Initial random tracers position
        self.tracers =  np.asarray(255.0 * np.random.random(
                self._tracersCount * 4), dtype=np.uint8, order='C')
        self.tracersRes = np.ceil(np.sqrt(self.tracers.size / 4))
        self.iTracers = np.arange(self.tracers.size / 4, dtype=np.float32)

        # Create all textures
        # Tracers position stored in texture 0
        self._currentTracersPos = Texture(
                data=self.tracers,
                width=self.tracersRes, height=self.tracersRes)
        # Initial random tracers position stored in texture 1
        self._nextTracersPos = Texture(
                data=self.tracers,
                width=self.tracersRes, height=self.tracersRes)
        self.fieldTexture = Texture(data=self._fieldAsRGB,
                width=self._fieldAsRGB.shape[1],
                height=self._fieldAsRGB.shape[0],
                filt=gl.GL_LINEAR)
        self.backgroundTexture = Texture(data=self.emptyPixels,
                width=self.w_width, height=self.w_height)
        self.screenTexture = Texture(data=self.emptyPixels,
                width=self.w_width, height=self.w_height)

        # VAOS
        values = np.zeros(int(self.tracers.size/4), [('a_index', 'f4', 1)])
        values['a_index'] = np.asarray(self.iTracers,
                dtype=np.float32, order='C')

        self._fieldTexture = Texture(data=self.modulus, dtype=gl.GL_FLOAT)

        ## VAO index
        self._vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, values, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
                ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Screen quad
        quad = np.zeros(4, dtype=[('vert', 'f4', 2),('tex', 'f4', 2)])

        quad['vert'] = np.array([
                [-1,  1],
                [1,   1],
                [1,  -1],
                [-1, -1]], np.float32)

        quad['tex'] = np.array([
                [0, 1],
                [1, 1],
                [1, 0],
                [0, 0]], np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], np.int32)

        self._vaoQuad = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vaoQuad)
        quadVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quadVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad, gl.GL_STATIC_DRAW)

        self.IBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2,   gl.GL_FLOAT, gl.GL_FALSE, 4 * 4,
                ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2,   gl.GL_FLOAT, gl.GL_FALSE, 4 * 4,
                ctypes.c_void_p(8))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.frameBuffer = gl.glGenFramebuffers(1)

    def draw(self):
        if self.drawField:
            self.drawFieldTexture(None, 1.0)
        self.fieldTexture.bind(0)
        ## Bind texture with random tracers position
        self._currentTracersPos.bind(1)

        self.drawScreen()

        self.updateTracers()

    def drawScreen(self):
        # Draw background texture and tracers on screen framebuffer texture
        self.setRenderingTarget(self.screenTexture)
        gl.glViewport(0, 0, self.w_width, self.w_height)

        self.drawTexture(self.backgroundTexture, self.fadeOpacity)
        self.drawTracers()
        self.resetRenderingTarget()

        # Draw the screen framebuffer texture on the monitor window
        # Enable blending to support drawing on top of an existing background
        #(e.g. a map)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.drawTexture(self.screenTexture, 1.0)
        gl.glDisable(gl.GL_BLEND)

        self.backgroundTexture , self.screenTexture = (
                self.screenTexture, self.backgroundTexture)

    def drawFieldTexture(self, texture, opacity):
        self._fieldTexture.bind()
        self.fieldProgram.bind()
        self.fieldProgram.setUniforms((
            ('gMap', 0),
            ('start', 1.0),
            ('gamma', 0.9),
            ('rot', -5.3),
            ('reverse', False),
            ('minSat', 0.2),
            ('maxSat', 5.0),
            ('minLight', 0.5),
            ('maxLight', 0.9),
            ('startHue', 240.0),
            ('endHue', -300.0),
            ('useHue', True),
            ))
        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)

        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
                          ctypes.c_void_p(0))
        self.fieldProgram.unbind()

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)
        self._fieldTexture.unbind()

    def drawTexture(self, texture, opacity):
        self.screenProgram.bind()
        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        texture.bind(2)
        self.screenProgram.setUniforms((
            ('u_screen', 2),
            ('u_opacity', opacity),
            ))

        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
                          ctypes.c_void_p(0))
        self.screenProgram.unbind()
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def drawTracers(self):
        gl.glBindVertexArray(self._vao)
        self.drawProgram.bind()

        self.drawProgram.setUniforms((
            ('u_field', 0),
            ('palette', bool(self.palette)),
            ('u_tracers', 1),
            ('MVP', self._MVP),
            ('pointSize', self.pointSize),
            # CubeHelix
            ('start', 1.0),
            ('gamma', 0.9),
            ('rot', -5.3),
            ('reverse', False),
            ('minSat', 0.2),
            ('maxSat', 5.0),
            ('minLight', 0.5),
            ('maxLight', 0.9),
            ('startHue', 240.0),
            ('endHue', -300.0),
            ('useHue', True),
            ('u_tracersRes', float(self.tracersRes)),
            ))
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glDrawArrays(gl.GL_POINTS, 0, int(self.tracers.size / 4))

        self.drawProgram.unbind()
        gl.glBindVertexArray(0)

    def updateTracers(self):
        self.setRenderingTarget(self._nextTracersPos)
        gl.glViewport(0, 0, int(self.tracersRes), int(self.tracersRes))

        self.updateProgram.bind()

        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        self.updateProgram.setUniforms((
            ('u_field', 0),
            ('u_tracers', 1),
            ('periodic', self.periodic),
            ('u_rand_seed', random.random()),
            ('u_speed_factor', self.speedFactor),
            ('u_drop_rate', self.dropRate),
            ('u_drop_rate_bump', self.dropRateBump),
            ('fieldScaling', self.fieldScaling),
            ('u_fieldRes', self._fieldAsRGB.shape),
            ))
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0))

        # Swap buffers
        self._currentTracersPos, self._nextTracersPos = (
                self._nextTracersPos, self._currentTracersPos)

        self.resetRenderingTarget()

