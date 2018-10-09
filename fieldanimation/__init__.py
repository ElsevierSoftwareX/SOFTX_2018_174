import os
import numpy as np
import ctypes
import OpenGL.GL as gl

# Local imports
from .shader import Shader
from .texture import Texture
from .__version__ import __version__

# Fix ramdom sequence seed
np.random.seed(123)

GLSLDIR = os.path.join(os.path.dirname(__file__ ), 'glsl')
WORKGROUP_SIZE = 32

#------------------------------------------------------------------------------
def glInfo():
    """ Return OpenGL information dict
        WARNING: OpenGL context MUST be initialized !!!

        Args:
            None

        Returns:
            OpenGL information dict
    """
    major = gl.glGetIntegerv(gl.GL_MAJOR_VERSION)
    minor = gl.glGetIntegerv(gl.GL_MINOR_VERSION)
    version = gl.glGetString(gl.GL_VERSION)
    vendor = gl.glGetString(gl.GL_VENDOR)
    renderer = gl.glGetString(gl.GL_RENDERER)
    glsl = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
    glversion = float("%d.%d" % (major, minor))
    retval = {
            'glversion': glversion,
            'version': version,
            'vendor': vendor,
            'renderer': renderer,
            'glsl': glsl,
            }
    if glversion >= 4.3:
        count = np.zeros(3, dtype=np.int32)
        size = np.zeros(3, dtype=np.int32)
        count[0] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0)[0]
        count[1] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1)[0]
        count[2] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2)[0]
        size[0] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0)[0]
        size[1] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1)[0]
        size[2] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2)[0]
        retval['maxComputeWorkGroupCount'] = count
        retval['maxComputeWorkGroupSize'] = size
    return retval

#------------------------------------------------------------------------------
def field2RGB(field):
    """ Return 2D field converted to uint8 RGB image (i.e. scaled in [0, 255])

        Args:
            field (:class:`numpy.ndarray`): (u, v) 2D vector field instance

        Returns:
            (
            rgb (:class:`numpy.ndarray`): uint8 RGB image,
            uMin (float): u min,
            uMax (float): u max,
            vMin (float): v min,
            vMax (float): v max,
            ) (tuple): Return value

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
    """ Return normalized modulus of 2D field image

        Returns:
            normalized modulus of 2D field image (i.e. scaled in [0, 1.])
    """
    modulus = np.flipud(np.hypot(field[:, :, 0], field[:,:,1]))
    return np.asarray(modulus/modulus.max(), np.float32)

#==============================================================================
class FieldAnimation(object):
    """ Field Animation with OpenGL


        1. draw the modulus of the vector field or a user defined image
            if requested;
        2. set a framebuffer texture (screen texture) as the main
            rendering target:
                (a) draw the background texture on the screen texture
                    with a fixed opacity;
                (b) decode the particles positions from the
                    currentTracersPosition texture and draw them on
                    the screen texture;
        3. set the rendering target to the active window;
        4. draw screen texture on the active window;
        5. swap screen texture and background texture;
        6. calculate the new particles positions
            (in the update shader) and encode them in the
            nextTracersPosition texture;
        7. swap nextTracersPosition texture and
            currentTracersPosition texture;
    """
    def __init__(self, width, height, field, computeSahder=False,
            image=None):
        """ Animate 2D vector field

            Args:
                width (int): width in pixels
                height (int): height in pixels
                field (np.ndarray): 2D vector field
                cs = True selects the compute shader version
                image = Optional background image
        """
        self.useComputeShader = computeSahder
        self.imageFileName = image
        # Parameters that can be changed later
        self.periodic = True
        self.drawField = False
        self.fadeOpacity = 0.996
        self.decayBoost = 0.01
        self.speedFactor = 0.25
        self.decay = 0.003
        self.palette = True
        self.color = (0.5, 1.0, 1.0)
        self.pointSize = 1.0
        self._tracersCount = 10000
        self.fieldScaling = 1.0

        # These are fixed
        self.w_width = width
        self.w_height = height

        # Since points are in [0, 1] a traslation and a scaling is needed on
        # the model matrix
        T = np.eye(4, dtype=np.float32)
        T[:, -1] = (-1., 1., 0, 1)
        S = (np.eye(4, dtype=np.float32)
                * np.array((2., -2., 1., 1.), dtype=np.float32))
        # Model transform matrix
        model = np.dot(T, S)
        # View matrix
        view = np.eye(4)

        # Projection matrix
        proj = np.eye(4)
        self.drawMVP = np.dot(model, np.dot(view, proj))
        self.fieldMVP = np.eye(4)

        # CubeHelix color palette parameters
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

        # Create Shader program and uniforms for the vector field
        self.fieldProgram = Shader(vertex='field.vert', fragment='field.frag',
                path=GLSLDIR)
        self.fieldProgram.addUniforms((('gMap', 'i'),
                ('MVP', 'mat4'),
                ) + cubeHelixParams)

        # Create Shader program and uniforms for the tracers
        self.drawProgram = Shader(vertex='draw.vert', fragment='draw.frag',
                path=GLSLDIR)
        self.drawProgram.addUniforms((
            ('MVP', 'mat4'),
            ('u_tracers', 'i'),
            ('u_tracersRes', 'f'),
            ('palette', 'b'),
            ('pointSize', 'f'),
            ('u_field', 'i'),
            ('u_fieldMin', '2f'),
            ('u_fieldMax', '2f')) + cubeHelixParams)

        # Create Shader program and uniforms for updating the screen
        self.screenProgram = Shader(vertex='quad.vert',
                fragment='screen.frag', path=GLSLDIR)
        self.screenProgram.addUniforms((
            ('u_screen', 'i'),
            ('u_opacity', 'f')))

        # Create image background shader
        if self.imageFileName:
            self.imageProgram = Shader(vertex='field.vert',
                    fragment='image.frag', path=GLSLDIR)
            self.imageProgram.addUniform('gMap', 'i')

        # Create Shader program and uniforms for updating the tracers position
        if self.useComputeShader:
            self.updateProgram = Shader(compute='update.comp', path=GLSLDIR)
        else:
            self.updateProgram = Shader(vertex='quad.vert',
                    fragment='update.frag', path=GLSLDIR)
        self.updateProgram.addUniforms((
                ('u_tracers', 'i'),
                ('u_field', 'i'),
                ('u_fieldRes', '2f'),
                ('u_fieldMin', '2f'),
                ('u_fieldMax', '2f'),
                ('u_rand_seed', 'f'),
                ('u_speed_factor', 'f'),
                ('u_decay', 'f'),
                ('u_decay_boost', 'f'),
                ('fieldScaling', 'f'),
                ('periodic', 'b')))

        # Set the vector field
        self.setField(field)
        self._initTracers()


    def setField(self, field):
        """ Set the 2D vector field. Must be called every time a new
            vetor field is selected.

            Args:
                field (np.ndarray): 2D vector field
        """
        # Automatic field scalking
        self.fieldScaling = self.speedFactor * 0.01 / field.max()

        # Prepare the data
        self._fieldAsRGB, uMin, uMax, vMin, vMax = field2RGB(field)
        # Compute field modulus
        self.modulus = modulus(field)
        # Set values that will not change
        self.drawProgram.bind()
        self.drawProgram.setUniform('u_fieldMin', (uMin, vMin))
        self.drawProgram.setUniform('u_fieldMax', (uMax, vMax))
        self.drawProgram.unbind()
        # Set values that will not change
        self.updateProgram.bind()
        self.updateProgram.setUniform('u_fieldMin', (uMin, vMin))
        self.updateProgram.setUniform('u_fieldMax', (uMax, vMax))
        self.updateProgram.unbind()
        self._initTracers()

    def setRenderingTarget(self, texture):
        """ Set texture as rendering target

            Args:
                texture (class:`texture` instance): 2D vector field
        """
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frameBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, texture.handle(), 0)

    def setSize(self, width, height):
        """ Set instance size. Must be called when the window is resized.

            Args:
                width (int): window width in pixels
                height (int): window height in pixels
        """
        self.w_width = width
        self.w_height = height
        self._initTracers()

    def resetRenderingTarget(self):
        """ Bind first (default) framebuffer and reset the viewport.
        """
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.w_width, self.w_height)

    @property
    def tracersCount(self):
        """ Return tracers count

            Returns:
                number of tracers
        """
        return  self._tracersCount

    @tracersCount.setter
    def tracersCount(self, value):
        """ Tracer count setter method, calls self._initTracers under
            the hood.

            Args:
                value (int): number of tracers to create
        """
        self._tracersCount = value
        self._initTracers()

    def _initTracers(self):
        """ Initialize the tracers positions
        """

        # Create a buffer for the tracers
        self.emptyPixels = np.zeros((self.w_width * self.w_height * 4),
                np.uint8)

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

        if self.imageFileName:
            self.imageTexture = Texture(data=self.imageFileName,
                    dtype=gl.GL_UNSIGNED_BYTE)
        self.modulusTexture = Texture(data=self.modulus, dtype=gl.GL_FLOAT)

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
        """ Render the OpenGL scene. This method is called automatically
            when the scene has to be rendered and is responsible for the
            animation.
        """
        if self.drawField:
            self.drawModulus(1.0)

        if self.imageFileName:
            self.drawImage()

        self.fieldTexture.bind(0)
        ## Bind texture with random tracers position
        self._currentTracersPos.bind(1)
        self.drawScreen()
        if self.useComputeShader:
            self.updateTracersCS()
        else:
            self.updateTracers()

    def drawScreen(self):
        """ Draw background texture and tracers on screen framebuffer texture
        """
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

    def drawModulus(self, opacity):
        """ Draw the modulus texture.

            Args:
                opacity (float): opacity (alpha) of the texture:
                    0 --> transparent
                    1 --> opaque
        """
        self.modulusTexture.bind()
        self.fieldProgram.bind()
        self.fieldProgram.setUniforms((
            ('gMap', 0),
            ('MVP', self.fieldMVP),
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
        self.modulusTexture.unbind()

    def drawImage(self):
        """ Draw an image texture in background.
        """
        self.imageProgram.bind()
        self.imageTexture.bind()
        self.imageProgram.setUniform('gMap', 0)
        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)

        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
                          ctypes.c_void_p(0))
        self.imageProgram.unbind()

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)
        self.imageTexture.unbind()

    def drawTexture(self, texture, opacity):
        """ Draw `texture` on the screen.

            Args:
                texture (:class:Texture): texture instance
                opacity (float): opacity (alpha) of the texture
                    0 --> transparent
                    1 --> opaque
        """
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
        texture.unbind()

    def drawTracers(self):
        """ Draw the tracers on the screen
        """
        gl.glBindVertexArray(self._vao)
        self.drawProgram.bind()

        self.drawProgram.setUniforms((
            ('u_field', 0),
            ('palette', bool(self.palette)),
            ('u_tracers', 1),
            ('MVP', self.drawMVP),
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
        """ Update tracers position using the fragment shader provided by
            the graphic card for computing.
        """
        self.setRenderingTarget(self._nextTracersPos)
        gl.glViewport(0, 0, int(self.tracersRes), int(self.tracersRes))

        self.updateProgram.bind()

        gl.glBindVertexArray(self._vaoQuad)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        self.updateProgram.setUniforms((
            ('u_field', 0),
            ('u_tracers', 1),
            ('periodic', self.periodic),
            ('u_rand_seed', np.random.random()),
            ('u_speed_factor', self.speedFactor),
            ('u_decay', self.decay),
            ('u_decay_boost', self.decayBoost),
            ('fieldScaling', self.fieldScaling),
            ('u_fieldRes', self._fieldAsRGB.shape),
            ))
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0))
        # Replace current tracers positions with the new ones
        self._currentTracersPos = self._nextTracersPos
        self.resetRenderingTarget()

    def updateTracersCS(self):
        """ Update tracers position using the compute shader provided by
            the graphic card for computing.
        """
        self.updateProgram.bind()
        self.updateProgram.setUniforms((
            ('u_field', 0),
            ('u_tracers', 1),
            ('periodic', self.periodic),
            ('u_rand_seed', np.random.random()),
            ('u_speed_factor', self.speedFactor),
            ('u_decay', self.decay),
            ('u_decay_boost', self.decayBoost),
            ('fieldScaling', self.fieldScaling),
            ('u_fieldRes', self._fieldAsRGB.shape),
            ))

        gl.glBindImageTexture(0, self._currentTracersPos._handle, 0,
                gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RGBA8)
        gl.glBindImageTexture(1, self._nextTracersPos._handle, 0,
                gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA8)

        gl.glDispatchCompute(int(self.tracersRes / WORKGROUP_SIZE),
                int(self.tracersRes / WORKGROUP_SIZE), 1)
        ## Lock memory access until image process ends
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        self._currentTracersPos = self._nextTracersPos

        self.resetRenderingTarget()
