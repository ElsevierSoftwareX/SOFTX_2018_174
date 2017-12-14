from glm import Vector3, Vector2
import math
import glfw

class Camera(object):

    STEP_SCALE = 1.0
    EDGE_STEP = 0.5
    MARGIN = 10

    def __init__(self, window, width, height, pos=Vector3(0., 0., 0.),
             target=Vector3(0., 0., 1.), up=Vector3(0., 1., 0.)):
        self._window = window
        self._width = width
        self._height = height
        self._pos = pos
        self._target = target
        self._up = up

        hTarget = Vector3(self._target.x, 0.0, self._target.z)
        hTarget.normalize()

        if hTarget.z >= 0.0:
            if hTarget.x >= 0.0:
                self._angleH = 360.0 - math.degrees(math.asin(hTarget.z))
            else:
                self._angleH = 180.0 + math.degrees(math.asin(hTarget.z))
        else:
            if hTarget.x >= 0.0:
                self._angleH = math.degrees(math.asin(-hTarget.z))
            else:
                self._angleH = 180.0 - math.degrees(math.asin(-hTarget.z))

        self._angleV = -math.degrees(math.asin(self._target.y))

        self._onUpperEdge = False
        self._onLowerEdge = False
        self._onLeftEdge = False
        self._onRightEdge = False
        self._mousePos = Vector2(width/2.0, height/2.0)
        self._onUpperEdge = False

    def setTarget(self, target):
        self._target = target

    def setUp(self, up):
        self._up = up

    def setPos(self, pos):
        self._pos = pos

    def onKeyboard(self, key):
        if key == glfw.KEY_UP:
            self._pos -= (self._target * Camera.STEP_SCALE)
        
        elif key == glfw.KEY_DOWN:
            self._pos += (self._target * Camera.STEP_SCALE)
        
        elif key == glfw.KEY_LEFT:
            left = self._target.cross(self._up)
            left.normalize()
            left *= Camera.STEP_SCALE
            self._pos -= left
        
        elif key == glfw.KEY_RIGHT:
            right = self._up.cross(self._target)
            right.normalize()
            right *= Camera.STEP_SCALE
            self._pos -= right
            
        elif key == glfw.KEY_PAGE_UP:
            self._pos.y -= Camera.STEP_SCALE
            
        elif key == glfw.KEY_PAGE_DOWN:
            self._pos.y += Camera.STEP_SCALE
    
    def onMouseScroll(self, dx, dy):
        """ dx is not  used"""
        self._pos += (self._target * Camera.STEP_SCALE)*dy

    def onMouse(self, x, y):

        xpos, ypos = glfw.get_cursor_pos(self._window)

        dx = x - xpos
        dy = y - ypos
        # dx = x - self._mousePos.x
        # dy = y - self._mousePos.y

        self._mousePos.x = x
        self._mousePos.y = y

        self._angleH += (dx / 20.0)
        self._angleV += (dy / 20.0)

        if dx == 0:
            if x <= Camera.MARGIN:
                self._onLeftEdge = True            
            elif x >= (self._width - Camera.MARGIN)  :
               self._onRightEdge = True
        else:
            self._onLeftEdge = False
            self._onRightEdge = False
        
        if dy == 0:
            if y <= Camera.MARGIN:
                self._onUpperEdge = True
            elif y >= (self._height - Camera.MARGIN):
                self._onLowerEdge = True
        else:
            self._onUpperEdge = False
            self._onLowerEdge = False
        
        self._update()
        
    def _update(self):
        vAxis = Vector3(0.0, 1.0, 0.0)

        # Rotate the view vector by the horizontal angle around the vertical axis
        view = Vector3(1.0, 0.0, 0.0)
        view.rotate(self._angleH, vAxis)
        view.normalize()

        # Rotate the view vector by the vertical angle around the horizontal axis
        hAxis = vAxis.cross(view)
        hAxis.normalize()
        view.rotate(self._angleV, hAxis)

        self._target = view.copy()
        self._target.normalize()

        self._up = self._target.cross(hAxis).copy()
        self._up.normalize()

    def onRender(self):
        needUpdate = False

        if self._onLeftEdge:
            self._angleH -= Camera.EDGE_STEP
            needUpdate = True

        elif self._onRightEdge:
            self._angleH += Camera.EDGE_STEP
            needUpdate = True

        if self._onUpperEdge:
            if self._angleV > -90.0:
                self._angleV -= Camera.EDGE_STEP
                needUpdate = True

        elif self._onLowerEdge:
            if self._angleV < 90.0:
               self._angleV += Camera.EDGE_STEP
               needUpdate = True

        if needUpdate:
            self._update()


        