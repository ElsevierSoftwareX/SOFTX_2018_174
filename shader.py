from OpenGL.error import GLError
import OpenGL.GL as gl
import os
import numpy as np

#==============================================================================
class Shader(object):
    """ Base shader class
    """
    def __init__(self, path=None, **kargs):

        self._uniforms = {}
        self._setters = {}
        self._sourceCode = {}
        # create the program handle
        self.program = gl.glCreateProgram()
        if not self.program:
            raise SystemExit("Shader creation failed: "
                    "Could not find valid memory location in constructor")
        # we are not linked yet
        self._linked = False

        for stype in kargs:
            shader = kargs[stype]
            if path is not None:
                shader = os.path.join(path, shader)
            if shader and os.path.isfile(shader):
                with open(shader) as f:
                    self._sourceCode[stype] = f.read()
            else:
                self._sourceCode[stype] = f.read()

            code = self._sourceCode[stype]
            if code:
                if stype == 'vertex':
                    self._build_shader(code, gl.GL_VERTEX_SHADER)
                elif stype == 'fragment':
                    self._build_shader(code, gl.GL_FRAGMENT_SHADER)
                elif stype == 'geometry':
                    self._build_shader(code, gl.GL_GEOMETRY_SHADER)
                elif stype == 'compute':
                    self._build_shader(code, gl.GL_COMPUTE_SHADER)
                else:
                    raise SystemExit("Unimplemented shader type '%s': "
                    "Giving up...")

        self._link()

    def addUniform(self, uniform, utype=None):
        """ Add a uniform variable to the shader.
            If utype is not none it defines the setter function.
            Valid utypes are i, b, f, 2f, 4fv
        """
        uniformLocation = gl.glGetUniformLocation(self.program, uniform)
        if uniformLocation == -1:
            print('Cannot find uniform %s' % uniform)
            return
        self._uniforms[uniform] = uniformLocation
        if utype is not None:
            if utype in "i b".split():
                self._setters[uniform] = gl.glUniform1i
            elif utype == 'f':
                self._setters[uniform] = gl.glUniform1f
            elif utype == '2f':
                self._setters[uniform] = gl.glUniform2f
            elif utype == 'mat4':
                self._setters[uniform] = gl.glUniformMatrix4fv
            else:
                raise SystemExit("Unimplemented utype '%s'" % utype)

    def addUniforms(self, uniformlist):
        for uniform in uniformlist:
            if isinstance(uniform, tuple):
                self.addUniform(*uniform)
            else:
                self.addUniform(uniform)

    def _build_shader(self, sourceCode, shader_type):
        """ Actual building of the shader
        """
        # create the shader handle
        shader = gl.glCreateShader(shader_type)
        if shader == 0:
            raise SystemExit("Shader creation failed: "
                    "Could not find valid memory location when adding shader")

        # Upload shader code
        gl.glShaderSource(shader, sourceCode)

        try:
            # compile the shader
            gl.glCompileShader(shader)
        except GLError as e:
            raise SystemExit(gl.glGetShaderInfoLog(shader))
        else:
            gl.glAttachShader(self.program, shader)

    def delete(self):
        programs = gl.glGetAttachedShaders(self.program)
        for item in programs:
            gl.glDetachShader(self.program, item)
            gl.glDeleteShader(item)
        gl.glDeleteProgram(self.program)

    def _link(self):
        """ Link the program
        """
        gl.glLinkProgram(self.program)
        # retrieve the link status

        if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            raise SystemExit(gl.glGetProgramInfoLog(self.program))
        else:
            gl.glValidateProgram(self.program)
            self._linked = True

        if not gl.glGetProgramiv(self.program, gl.GL_VALIDATE_STATUS):
            raise SystemExit(gl.glGetProgramInfoLog(self.program))

    def bind(self):
        """ Bind the program, i.e. use it
        """
        gl.glUseProgram(self.program)

    def unbind(self):
        """ Unbind whatever program is currently bound - not necessarily this
            program, so this should probably be a class method instead.
        """
        gl.glUseProgram(0)

    def setUniform(self, name, value):
        if isinstance(value, (list, tuple)):
            self._setters[name](self._uniforms.get(name), *value)
        elif isinstance(value, np.ndarray) and value.shape == (4, 4):
            self._setters[name](self._uniforms.get(name), 1, True, value)
        else:
            try:
                self._setters[name](self._uniforms.get(name), value)
            except:
                raise SystemExit("Wrong setter function for '%s'" % name)

    def setUniforms(self, nameandvalue):
        for name, value in nameandvalue:
            self.setUniform(name, value)

    def code(self, shader='v', lineno=False):
        """ Return shader code.

            Keyword Args:
                shader (string): v for vertex, f for fragment, g for geometry
                lineno (boolean): add line numbers to code

            Returns:
                shader source code
        """
        if lineno:
            code = ''
            for lineno,line in enumerate(self._vertex_code.split('\n')):
                code += '%3d: ' % (lineno + 1) + line + '\n'
            return code
        else:
            return self._vertex_code

    def vertex_code(self, lineno=False):
        """ Return vertex shader code with optional line numbers
        """
        if lineno:
            code = ''
            for lineno,line in enumerate(self._vertex_code.split('\n')):
                code += '%3d: ' % (lineno + 1) + line + '\n'
            return code
        else:
            return self._vertex_code

    def fragment_code(self, lineno=False):
        """ Return fragment shader code with optional line numbers
        """
        if lineno:
            code = ''
            for lineno,line in enumerate(self._fragment_code.split('\n')):
                code += '%3d: ' % (lineno + 1) + line + '\n'
            return code
        else:
            return self._fragment_code

    def geometry_code(self, lineno=False):
        """ Return geometry shader code with optional line numbers
        """
        if self._geometry_code:
            if lineno:
                code = ''
                for lineno,line in enumerate(self._geometry_code.split('\n')):
                    code += '%3d: ' % (lineno + 1) + line + '\n'
                return code
            else:
                return self._fragment_code

