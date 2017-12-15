import glfw
import OpenGL.GL as gl
from pipeline import Pipeline
from camera import Camera


actionMap = {glfw.PRESS: 'press',
             glfw.RELEASE: 'release',
             glfw.REPEAT: 'repeat'}

def errorCallback(error, description):
    print('Error %s, %s' % (error, description))

class glfwApp(object):
    
    KEY_G = glfw.KEY_G
    PRESS = glfw.PRESS
    RELEASE = glfw.RELEASE
    
    def __init__(self, title='', width=800, height=600):

        self._width = width
        self._height = height
        self._title = title

        glfw.set_error_callback(errorCallback)

        if not glfw.init():
            raise SystemExit("Error initializing GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)

        self._createWindow()

        self._camera = Camera(self._window, self._width, self._height)
        self._pipeline = Pipeline()
        self._pipeline.setCamera(self._camera)

        self.setInput()

    def camera(self):
        return self._camera

    def pipeline(self):
        return self._pipeline

    def _createWindow(self):
        self._window = glfw.create_window(self._width, self._height, self._title, None, None)

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
        #self._camera.onKeyboard(key)
        # print(key, scancode, actionMap.get(action), mode)

    def onMouseMove(self, window, x, y):
        if glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) or glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT):
            return
        self._camera.onMouse(x, y)

    def onMouseWheel(self, window, dx, dy):
        self._camera.onMouseScroll(dx, dy)


    def window(self):
        return self._window

    def setInput(self):

        glfw.set_key_callback(self._window, self.onKeyboard)
        glfw.set_cursor_pos_callback(self._window, self.onMouseMove)
        # glfw.set_mouse_button_callback(self._window, None)
        glfw.set_scroll_callback(self._window, self.onMouseWheel)


    def run(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # NOTE: Must be implemented in the render method
        # gl.glFrontFace(gl.GL_CW)
        # gl.glCullFace(gl.GL_BACK)
        # gl.glEnable(gl.GL_CULL_FACE)
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
        # return

if __name__ == "__main__":
    app = glfwApp('glfwApp')
    app.run()

