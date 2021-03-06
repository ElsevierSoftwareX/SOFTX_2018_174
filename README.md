# FIELDANIMATION #
FIELDANIMATION a Python package to animate 2D vector fields.

### What is it? ###
Field Animation is a Python package for the visualization of 2D vector fields
that exploits the capabilities of modern graphic cards. Vector fields are
represented through particles that move along the flow lines of the field at
a speed and color proportional to its modulus in an OpenGL window.

Interactive control of speed, color and number of animated particles is
possible as the whole rendering process happens in the GPU in real time.
A background image can be shown to add information for the interpretation of
the results.

Highlights:

* Animation parameters can be changed in real time

* The OpenGL window can be embedded in any Python application.

### Where is the documentation? ###

[Here!](https://bvidmar.bitbucket.io/fieldanimation/)

### Who do I talk to? ###

* [Nicola Creati](mailto:ncreati@inogs.it)

* [Roberto Vidmar](mailto:rvidmar@inogs.it)

### NOTE: ###
    To run the example application in the examples directory uncomment
    the first threee requirements in *requirements.txt* or manually install
    Pillow, imgui and glfw.
