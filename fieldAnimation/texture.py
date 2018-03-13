import OpenGL.GL as gl

class Texture(object):
    def __init__(self, data=None,  width=None, height=None,
                filt=gl.GL_NEAREST, dtype=gl.GL_UNSIGNED_BYTE):
        """ Texture object.
            If data is None an empty texture will be created
        """

        self._data = data

        # format of texture object
        if self._data.ndim > 2 and self._data.shape[2] == 3:
            self._format = gl.GL_RGB
        else:
            self._format = gl.GL_RGBA

        if dtype == gl.GL_FLOAT:
            self._format = gl.GL_R32F
            self._image_format = gl.GL_RED
            self._pname = gl.GL_REPEAT
            self._filter_min = gl.GL_LINEAR
            self._filter_mag = gl.GL_LINEAR
        else:
            self._pname = gl.GL_CLAMP_TO_EDGE
            ## Filtering mode if texture pixels < screen pixels
            self._filter_min = filt
            ## Filtering mode if texture pixels > screen pixels
            self._filter_mag = filt
            self._image_format = self._format

        # Create the Texture
        self._handle = gl.glGenTextures(1)
        if self._data is not None:
            if width is None or height is None:
                try:
                    height, width, bands = self._data.shape
                except ValueError:
                    height, width = self._data.shape

        # Bind texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)
        # Avoid banding if row length is odd
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # Set Texture wrap and filter modes
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                self._pname);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                self._pname);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                self._filter_min);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                self._filter_mag);
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self._format, width, height,
                0, self._image_format, dtype, self._data)
        # Unbind texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

    def bind(self, texUnit=0):
        gl.glActiveTexture(gl.GL_TEXTURE0 + texUnit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)

    def handle(self):
        return self._handle

    def unbind(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
