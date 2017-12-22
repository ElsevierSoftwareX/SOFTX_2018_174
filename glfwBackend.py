import glfw
import OpenGL.GL as gl


actionMap = {glfw.PRESS: 'press',
             glfw.RELEASE: 'release',
             glfw.REPEAT: 'repeat'}

def errorCallback(error, description):
    print('Error %s, %s' % (error, description))

class glfwApp(object):

    KEY_G = glfw.KEY_G
    KEY_N = glfw.KEY_N
    KEY_F = glfw.KEY_F
    PRESS = glfw.PRESS
    RELEASE = glfw.RELEASE

    def __init__(self, title='', width=800, height=600):

        self._width = width
        self._height = height
        self._title = title

        glfw.set_error_callback(errorCallback)

        if not glfw.init():
            raise SystemExit("Error initializing GLFW")
        glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)
        self._createWindow()
        self.setInput()

    def _createWindow(self):
        self._window = glfw.create_window(self._width, self._height,
                self._title, None, None)

        if not self._window:
            glfw.terminate()
            raise SystemExit

        glfw.make_context_current(self._window)

    def onKeyboard(self, window, key, scancode, action, mode):
        """
        :param window:
        :param key:
        :param scancode:
        :param action: PRESS, RELEASE, REPEAT
        :param mode: modifiers
        :return:
        """

        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(self._window, 1)
        glfw.poll_events()

    def window(self):
        return self._window

    def title(self):
        """ Return window title
        """
        return self._title

    def setTitle(self, title):
        glfw.set_window_title(self._window, title)

    def setInput(self):
        glfw.set_key_callback(self._window, self.onKeyboard)

    def run(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        while not glfw.window_should_close(self._window):
            self.renderScene()
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        self.close()

    def close(self):
        glfw.destroy_window(self._window)
        glfw.terminate()

    def renderScene(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

if __name__ == "__main__":
    app = glfwApp('glfwApp')
    app.run()

