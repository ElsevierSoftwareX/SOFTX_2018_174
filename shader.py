from OpenGL.error import GLError
import OpenGL.GL as gl
import ctypes
import os
import numpy as np
import glm

#==============================================================================
class Shader(object):
    ''' Base shader class. '''
    def __init__(self, vertex=None, fragment=None,
                geometry=None):

        self._uniforms = {}

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
            print("Shader creation failed: "
                    "Could not find valid memory location in constructor")
            raise SystemExit

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
        #self._uniforms[name] = Uniform(self._shader, name, gtype, None)

    def addUniform(self, uniform):
        uniformLocation = gl.glGetUniformLocation(self._program, uniform)
        if uniformLocation == -1:
            print('Cannot find uniform %s' % uniform)
            return
        self._uniforms[uniform] = uniformLocation


    def _build_shader(self, strings, shader_type):
        ''' Actual building of the shader '''

        count = len(strings)
        # if we have no source code, ignore this shader
        if count < 1:
            return

        # create the shader handle
        shader = gl.glCreateShader(shader_type)
        if shader == 0:
            print("Shader creation failed: Could not find valid memory location when adding shader")
            raise SystemExit

        # Upload shader code
        gl.glShaderSource(shader, strings)

        try:
            # compile the shader
            gl.glCompileShader(shader)
        except GLError as e:
            print(gl.glGetShaderInfoLog(shader))
            raise SystemExit
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

    def updateUniforms(self, worldMatrix, projecttedMatrix):
        pass

    def _link(self):
        ''' Link the program '''

        gl.glLinkProgram(self._program)
        # retrieve the link status
        temp = ctypes.c_int(0)
#        gl.glGetProgramiv(self._program, gl.GL_LINK_STATUS, ctypes.byref(temp))

        if not gl.glGetProgramiv(self._program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(self._program))
            raise SystemExit
        else:
            gl.glValidateProgram(self._program)
            self._linked = True

        if not gl.glGetProgramiv(self._program, gl.GL_VALIDATE_STATUS):
            print(gl.glGetProgramInfoLog(self._program))
            raise SystemExit

    def bind(self):
        ''' Bind the program, i.e. use it. '''
        #print 'aaaa',gl.glIsProgram( self.handle )

        gl.glUseProgram(self._program)


    def unbind(self):
        ''' Unbind whatever program is currently bound - not necessarily this
            program, so this should probably be a class method instead. '''
        gl.glUseProgram(0)

    def setUniformi(self, name, value):
        gl.glUniform1i(self._uniforms.get(name), value)

    def setUniformf(self, name, value):
        gl.glUniform1f(self._uniforms.get(name), value)

    def setUniform2f(self, name, v1, v2):
        variable = self._uniforms.get(name)
        if variable != -1:
            gl.glUniform2f(self._uniforms.get(name), v1, v2)

    def setUniform(self, name, value):
        if isinstance(value, glm.Vector3):
            gl.glUniform3fv(self._uniforms.get(name), 1, value)
        elif isinstance(value, glm.Matrix4f):
            gl.glUniformMatrix4fv(self._uniforms.get(name), 1, True, value)
        elif isinstance(value, int):
            gl.glUniform1i(self._uniforms.get(name), value)
        elif isinstance(value, float):
            gl.glUniform1f(self._uniforms.get(name), value)
        else:
            print('Unknown uniform type %s=%s' %(name, value))

    def vertex_code(self, lineno=False):
        if lineno:
            code = ''
            for lineno,line in enumerate(self._vertex_code.split('\n')):
                code += '%3d: ' % (lineno+1) + line + '\n'
            return code
        else:
            return self._vertex_code

    def fragment_code(self, lineno=False):
        if lineno:
            code = ''
            for lineno,line in enumerate(self._fragment_code.split('\n')):
                code += '%3d: ' % (lineno+1) + line + '\n'
            return code
        else:
            return self._fragment_code

    def geometry_code(self, lineno=False):
        if self._geometry_code:
            if lineno:
                code = ''
                for lineno,line in enumerate(self._geometry_code.split('\n')):
                    code += '%3d: ' % (lineno+1) + line + '\n'
                return code
            else:
                return self._fragment_code

