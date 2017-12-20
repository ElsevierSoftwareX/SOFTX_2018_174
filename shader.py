from OpenGL.error import GLError
import OpenGL.GL as gl
import ctypes
import os
import numpy as np
import glm

#==============================================================================
class Shader(object):
    """ Base shader class
    """
    def __init__(self, vertex=None, fragment=None, geometry=None):

        self._uniforms = {}
        self._setters = {}

        if os.path.isfile(vertex):
            with open(vertex) as f:
                self._vertex_code = f.read()
        else:
            self._vertex_code = vertex

        if isinstance(fragment, (list, tuple)):
            self._fragment_code = ''
            for fn in fragment:
                with open(fn) as f:
                    self._fragment_code += f.read()
        elif os.path.isfile(fragment):
            with open(fragment) as f:
                self._fragment_code = f.read()
        else:
            self._fragment_code = fragment

        if geometry:
            if os.path.isfile(geometry):
                with open(geometry) as f:
                    self._geometry_code = f.read()
            else:
                self._geometry_code = geometry

        # create the program handle
        self._program = gl.glCreateProgram()
        if not self._program:
            raise SystemExit("Shader creation failed: "
                    "Could not find valid memory location in constructor")

        # we are not linked yet
        self._linked = False

        # create the vertex shader
        self._build_shader(self._vertex_code, gl.GL_VERTEX_SHADER)

        # create the fragment shader
        self._build_shader(self._fragment_code, gl.GL_FRAGMENT_SHADER)

        if geometry:
            # create the geometry shader
            self._build_shader(self._geometry_code, gl.GL_GEOMETRY_SHADER)

        self._link()

    def addUniform(self, uniform, utype=None):
        """ Add a uniform variable to the shader.
            If utype is not none it defines the setter function.
            Valid utypes are i, b, f, 2f, 4fv
        """
        uniformLocation = gl.glGetUniformLocation(self._program, uniform)
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
            elif utype == '4fv':
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
        if len(sourceCode) < 1:
            # if we have no source code, ignore this shader
            return

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
            gl.glAttachShader(self._program, shader)

    def delete(self):
        programs = gl.glGetAttachedShaders(self._program)
        for item in programs:
            gl.glDetachShader(self._program, item)
            gl.glDeleteShader(item)
        gl.glDeleteProgram(self._program)

    def program(self):
        return self._program

    def _link(self):
        """ Link the program
        """
        gl.glLinkProgram(self._program)
        # retrieve the link status
        temp = ctypes.c_int(0)

        if not gl.glGetProgramiv(self._program, gl.GL_LINK_STATUS):
            raise SystemExit(gl.glGetProgramInfoLog(self._program))
        else:
            gl.glValidateProgram(self._program)
            self._linked = True

        if not gl.glGetProgramiv(self._program, gl.GL_VALIDATE_STATUS):
            raise SystemExit(gl.glGetProgramInfoLog(self._program))

    def bind(self):
        """ Bind the program, i.e. use it
        """
        gl.glUseProgram(self._program)

    def unbind(self):
        """ Unbind whatever program is currently bound - not necessarily this
            program, so this should probably be a class method instead.
        """
        gl.glUseProgram(0)

    def setUniform(self, name, value):
        if isinstance(value, (list, tuple)):
            self._setters[name](self._uniforms.get(name), *value)
        elif isinstance(value, glm.Matrix4f):
            self._setters[name](self._uniforms.get(name), 1, True, value)
        else:
            try:
                self._setters[name](self._uniforms.get(name), value)
            except:
                raise SystemExit("Wrong setter function for '%s'" % name)

    def setUniforms(self, nameandvalue):
        for name, value in nameandvalue:
            self.setUniform(name, value)

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

