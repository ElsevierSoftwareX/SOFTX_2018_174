import os
import sys
import argparse
import numpy as np
from PIL import Image
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Gui
import tkinter as tk

import OpenGL.GL as gl
from glfwBackend import glfwApp
import imgui
from imgui.integrations.glfw import GlfwRenderer

# Local imports
from fieldAnimation import FieldAnimation


#------------------------------------------------------------------------------
def E(q, r0, x, y):
    """ Return the electric field vector E=(Ex, Ey) due to charge q at r0.
    """
    den = np.hypot(x - r0[0], y - r0[1]) ** 1.5
    return q * (x - r0[0]) / den, q * (y - r0[1]) / den

#------------------------------------------------------------------------------
def createField(m=64, n=64, eq=E, charges=1):
    """ Return n x m x 2 2D array with Electric field values
        generated by a (2 * charges) electric charges configuration
        Electric field follows equation eq.
    """
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, m)
    X, Y = np.meshgrid(x, y)

    # Create a multipole with nq charges of alternating sign, equally spaced
    # on the unit circle.
    nq = 2 ** int(charges)
    charges = []
    for i in range(nq):
        q = i % 2 * 2 - 1
        charges.append((q,
                (np.cos(2 * np.pi * i / nq), np.sin(2 * np.pi * i / nq))))

    Ex, Ey = np.zeros((m, n)), np.zeros((m, n))
    for charge in charges:
        ex, ey = eq(*charge, x=X, y=Y)
        Ex += ex
        Ey += ey
    return np.dstack((Ex, Ey))


#------------------------------------------------------------------------------
def userInterface(renderer, graphicItem):
    """ Control graphicItem parameters interactively
    """
    if not renderer:
        return

    renderer.process_inputs()
    imgui.new_frame()
    imgui.begin('Controls', closable=True,
            flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
    # Speed Rate
    speedChanged, speed = imgui.slider_float('Speed',
        graphicItem.speedFactor(), 0, 10.0)
    if speedChanged:
        graphicItem.setSpeedFactor(speed)

    # Drop Rate
    dropChanged, dropRate = imgui.slider_float('Drop rate',
        graphicItem.dropRate(), 0, 0.1)
    if dropChanged:
        graphicItem.setDropRate(dropRate)

    # Palette
    r, g, b = graphicItem.color()
    colorChanged, color = imgui.color_edit3('Color', r, g, b)
    if colorChanged:
        graphicItem.setColor(color)
    imgui.same_line()
    clicked, palette = imgui.checkbox("Palette", graphicItem.palette())
    if clicked:
        graphicItem.setPalette(palette)

    # Point size
    pointSizeChanged, pointSize = imgui.input_int("Point size",
        graphicItem.pointSize(), 1, 1, 1)
    if pointSizeChanged:
        if pointSize > 5:
            pointSize = 5
        elif pointSize < 1:
            pointSize = 1
        graphicItem.setPointSize(pointSize)

    # Number of Points
    pointsCountChanged, pointsCount = imgui.drag_int("Number of "
        "Points", graphicItem.pointsCount(), 4096.0, 64, 10000000)
    if pointsCountChanged:
        graphicItem.setPointsCount(pointsCount)

    imgui.end()
    imgui.render()

#==============================================================================
class GLApp(glfwApp):
    def __init__(self, title, width, height, field, options):
        super(GLApp, self).__init__(title, width, height)

        if options.gui:
            self._renderer = GlfwRenderer(self.window())
        else:
            self._renderer = None

        # Add Field Animation overlay
        self._fa = FieldAnimation(width, height, field)

    def renderScene(self):
        super(GLApp, self).renderScene()
        self._fa.draw()
        userInterface(self._renderer, self._fa)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="\nField Animation example",
            add_help=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog=os.path.basename(sys.argv[0]))

    parser.add_argument('-f', '--field', choices="epole wind".split(),
            default="wind",
            help=("Choose field to animate ")
            )

    parser.add_argument('-g', '--gui', action='store_true', default=False,
            help=("Add gui control window ")
            )

    options = parser.parse_args(sys.argv[1:])

    if options.field == 'wind':
        field = np.load("wind_2016-11-20T00-00Z.npy")
    elif options.field == 'epole':
        field = createField()

    app = GLApp('Field Animation', 360 * 3, 180 * 3, field, options)
    app.run()
