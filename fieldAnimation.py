from __future__ import division
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

GLSL = 'glsl'

#------------------------------------------------------------------------------
def makeProgram(vertexfile, fragmentfile):
    return Shader(os.path.join(GLSL, vertexfile),
            os.path.join(GLSL, fragmentfile))

#==============================================================================
class FieldAnimation(object):
    """ Field Animation plot class
    """
    def __init__(self, width, height, field=None):
        """ Animate 2D `field`
        """
        self.w_width = width
        self.w_height = height

        self.emptyPixels = np.zeros((width * height * 4), np.uint8)
        # Convert 2D field to RGB image and scale values in [0, 255]
        rows = field.shape[0]
        cols = field.shape[1]
        u = field[:, :, 0]
        v = field[:, :, 1]
        self._uMin = u.min()
        self._uMax = u.max()
        self._vMin = v.min()
        self._vMax = v.max()
        self._rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        self._rgb[:, :, 0] = 255 * (u - self._uMin) / (
                self._uMax - self._uMin)
        self._rgb[:, :, 1] = 255 * (v - self._vMin) / (
                self._vMax - self._vMin)

        self._fadeOpacity = 0.996
        self._dropRateBump = 0.01
        self._speedFactor = 0.25
        self._speedFactor = 0.5
        self._dropRate = 0.003
        self._palette = True
        self._color = (1.0, 1.0, 1.0)
        self._pointSize = 1.0
        self._pointsCount = 65536

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

        self.drawProgram = makeProgram('draw.vert', 'draw.frag')
        self.drawProgram.addUniform('MVP')
        self.drawProgram.addUniform('u_particles')
        self.drawProgram.addUniform('u_particles_res')
        # CubeHelix
        self.drawProgram.addUniform('start')
        self.drawProgram.addUniform('gamma')
        self.drawProgram.addUniform('rot')
        self.drawProgram.addUniform('reverse')
        self.drawProgram.addUniform('minSat')
        self.drawProgram.addUniform('maxSat')
        self.drawProgram.addUniform('minLight')
        self.drawProgram.addUniform('maxLight')
        self.drawProgram.addUniform('startHue')
        self.drawProgram.addUniform('endHue')
        self.drawProgram.addUniform('useHue')
        self.drawProgram.addUniform('palette')
        self.drawProgram.addUniform('pointSize')

        self.drawProgram.addUniform('u_wind')
        self.drawProgram.addUniform('u_wind_min')
        self.drawProgram.addUniform('u_wind_max')
        # self.drawProgram.addUniform('u_color_ramp')

        self.screenProgram = makeProgram('quad.vert', 'screen.frag')
        self.screenProgram.addUniform('u_screen')
        self.screenProgram.addUniform('u_opacity')

        self.updateProgram = makeProgram('quad.vert', 'update.frag')
        self.updateProgram.addUniform('u_particles')
        self.updateProgram.addUniform('u_wind')
        self.updateProgram.addUniform('u_wind_res')
        self.updateProgram.addUniform('u_wind_min')
        self.updateProgram.addUniform('u_wind_max')
        self.updateProgram.addUniform('u_rand_seed')
        self.updateProgram.addUniform('u_speed_factor')
        self.updateProgram.addUniform('u_drop_rate')
        self.updateProgram.addUniform('u_drop_rate_bump')

        self.build()

    def bindTexture(self, texture):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frameBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, texture.handle(), 0)

    def unbindTexture(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def speedFactor(self):
        return self._speedFactor

    def setSpeedFactor(self, value):
        self._speedFactor = value

    def dropRate(self):
        return self._dropRate

    def setDropRate(self, value):
        self._dropRate = value

    def pointSize(self):
        return self._pointSize

    def setPointSize(self, value):
        self._pointSize = value

    def pointsCount(self):
        return  self._pointsCount

    def setPointsCount(self, value):
        self._pointsCount = value
        self.build()

    def palette(self):
        return self._palette

    def setPalette(self, value):
        self._palette = value

    def color(self):
        return self._color

    def setColor(self, value):
        self._color = value

    def build(self):
        # Initial random particles position
        self.particleState =  np.asarray(255.0 * np.random.random(
                self.pointsCount()), dtype=np.uint8, order='C')
        self.num_particles = self.particleState.size/4
        self.particlesRes = np.ceil(np.sqrt(self.num_particles))
        self.particleIndices = np.arange(self.num_particles, dtype=np.float32)

        # Create all textures
        # Particles position stored in texture 0
        self._particleStateTexture0 = Texture(
                data=self.particleState,
                width=self.particlesRes, height=self.particlesRes)
        # Initial random particles position stored in texture 1
        self._particleStateTexture1 = Texture(
                data=self.particleState,
                width=self.particlesRes, height=self.particlesRes)
        self.windTexture = Texture(data=self._rgb,
                width=self._rgb.shape[1], height=self._rgb.shape[0],
                filt=gl.GL_LINEAR)
        self.backgroundTexture = Texture(data=self.emptyPixels,
                width=self.w_width, height=self.w_height)
        self.screenTexture = Texture(data=self.emptyPixels,
                width=self.w_width, height=self.w_height)

        # VAOS
        values = np.zeros(int(self.num_particles), [('a_index', 'f4', 1)])
        values['a_index'] = np.asarray(self.particleIndices,
                dtype=np.float32, order='C')

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
        self.windTexture.bind(0)
        # Bind texture with random particles position
        self._particleStateTexture0.bind(1)

        self.drawScreen()

        self.updateParticles()

    def drawScreen(self):

        self.bindTexture(self.screenTexture)
        gl.glViewport(0, 0, self.w_width, self.w_height)
        self.drawTexture(self.backgroundTexture, self._fadeOpacity)
        self.drawParticles()
        self.unbindTexture()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.drawTexture(self.screenTexture, 1.0)
        gl.glDisable(gl.GL_BLEND)

        self.backgroundTexture , self.screenTexture = (
                self.screenTexture, self.backgroundTexture)

    def drawTexture(self, texture, opacity):
        self.screenProgram.bind()
        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        texture.bind(2)
        self.screenProgram.setUniformi('u_screen', 2)
        self.screenProgram.setUniformf('u_opacity', opacity)

        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
                          ctypes.c_void_p(0))
        self.screenProgram.unbind()
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def drawParticles(self):
        gl.glBindVertexArray(self._vao)
        self.drawProgram.bind()

        # self.colorRampTexture.bind(2)

        self.drawProgram.setUniformi('u_wind', 0)
        self.drawProgram.setUniformi('u_particles', 1)
        self.drawProgram.setUniform('MVP', self._MVP)
        self.drawProgram.setUniformi('palette', bool(self.palette()))
        self.drawProgram.setUniformf('pointSize', self.pointSize())
        # self.drawProgram.setUniformi('u_color_ramp', 2)

        # CubeHelix
        self.drawProgram.setUniformf('start', 1.0)
        self.drawProgram.setUniformf('gamma', 0.9)
        self.drawProgram.setUniformf('rot', -5.3)
        self.drawProgram.setUniformi('reverse', False)
        self.drawProgram.setUniformf('minSat', 0.2)
        self.drawProgram.setUniformf('maxSat', 5.0)
        self.drawProgram.setUniformf('minLight', 0.5)
        self.drawProgram.setUniformf('maxLight', 0.9)
        self.drawProgram.setUniformf('startHue', 240.0)
        self.drawProgram.setUniformf('endHue', -300.0)
        self.drawProgram.setUniformi('useHue', True)

        self.drawProgram.setUniformf('u_particles_res',
                float(self.particlesRes))
        self.drawProgram.setUniform2f('u_wind_max', self._uMax, self._vMax)
        self.drawProgram.setUniform2f('u_wind_min', self._uMin, self._vMin)

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)

        gl.glDrawArrays(gl.GL_POINTS, 0, int(self.num_particles))

        self.drawProgram.unbind()
        gl.glBindVertexArray(0)

    def updateParticles(self):
        self.bindTexture(self._particleStateTexture1)
        gl.glViewport(0, 0, int(self.particlesRes), int(self.particlesRes))

        self.updateProgram.bind()

        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)

        self.updateProgram.setUniformi('u_wind', 0)
        self.updateProgram.setUniformi('u_particles', 1)
        self.updateProgram.setUniformf('u_rand_seed', random.random())

        self.updateProgram.setUniform2f('u_wind_res',
            self._rgb.shape[0], self._rgb.shape[1])
        self.updateProgram.setUniform2f('u_wind_min', self._uMin, self._vMin)
        self.updateProgram.setUniform2f('u_wind_max', self._uMax, self._vMax)

        self.updateProgram.setUniformf('u_speed_factor', self._speedFactor)
        self.updateProgram.setUniformf('u_drop_rate', self._dropRate)
        self.updateProgram.setUniformf('u_drop_rate_bump', self._dropRateBump)

        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0))

        self._particleStateTexture0 ,self._particleStateTexture1 = \
            (self._particleStateTexture1 ,self._particleStateTexture0)

        self.unbindTexture()

