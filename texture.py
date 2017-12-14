import OpenGL.GL as gl
from PIL import Image
import numpy as np

class Texture(object):
    def __init__(self, filename='', data=None,  width=None, height=None, filt=gl.GL_NEAREST):

        if filename:
            img = Image.open(filename)
            img = img.convert('RGBA')
            data = np.asarray(img, dtype=np.uint8)
        if (filename and data is not None) or (filename is None and data is None):
            print('Set filename or data not both!')
            raise SystemExit

        self._data = data
        self._filename = filename

        # format of texture object
        if data.ndim > 2 and data.shape[2] == 3:
            self._format = gl.GL_RGB
        else:
            self._format = gl.GL_RGBA


        # format of loaded image
        self._image_format = self._format

        # Texture configuration
        ## Filtering mode if texture pixels < screen pixels
        self._filter_min = filt
        ## Filtering mode if texture pixels > screen pixels
        self._filter_max = filt

        self._handle = gl.glGenTextures(1)
        if self._data is not None:
            self._generate(self._data, width, height)

    def handle(self):
        return self._handle

    def _generate(self, data, width=None, height=None):

        if width is None or height is None:
            try:
                w, h, bands = data.shape
            except:
                w, h = data.shape
        else:
            w = width
            h = height

        # Create the Texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)

        #gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # Set Texture wrap and filter modes
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self._filter_min);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self._filter_max);
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self._format, w, h, 0, self._image_format, gl.GL_UNSIGNED_BYTE, data)
        # Unbind texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

    def bind(self, texUnit=0):
        gl.glActiveTexture(gl.GL_TEXTURE0+texUnit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._handle)

    def unbind(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
