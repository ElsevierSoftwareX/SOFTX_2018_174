.. _PyOpenGL: http://pyopengl.sourceforge.net/

.. _numpy:  http://www.numpy.org

********************************************
FieldAnimation  Python Package Documentation
********************************************

Foreword
########
FieldAnimation has been developed by two programmers working in the 
`Aerial Operations <http://www.inogs.it/en/content/aerial-operations>`_
group of the IRI Research Section at
`OGS - Istituto Nazionale di Oceanografia e di
Geofisica Sperimentale <http://www.inogs.it>`_.

Python is their favourite programming language since 2006.

The authors:

**Nicola Creati, Roberto Vidmar**

.. image:: images/creati.jpg
   :height: 134 px
   :width:  100 px 
.. image:: images/vidmar.jpg
   :height: 134 px
   :width:  100 px 

What is FieldAnimation?
#######################
It is a Python package to represent vector fields through particles that
move along the flow lines of the field at a speed and color proportional
to its modulus.

A background image can be shown to add information for the interpretation
of the results.

An **example application** with interactive control of speed,
color and number of animated particles is available in the examples direcory.

FieldAnimation in a nutshell
############################
  * create a :class:`fieldanimation.FieldAnimation` instance
  * call its :class:`fieldanimation.FieldAnimation.draw` method in
    the main rendering loop


Dependencies
############
It relies on PyOpenGL_ and numpy_. The rendering of the OpenGL image must
be carried out by a library that handles windows, input and events like
`GLFW <http://www.glfw.org/>`_ or
`PyQt <https://riverbankcomputing.com/software/pyqt/>`_.

More
####
.. toctree::
  :maxdepth: 3
  :numbered:

  FieldAnimation Package Reference: <fieldanimation>
  An example application <app>
  How it works <code_design>


.. warning::
   This code has been tested *only* on Linux (Ubuntu 18.04.1 LTS and
   Ubuntu 17.10) but should work also on Mac and Windows.

.. warning::
   This is work in progress!

Indices and Tables
##################
* :ref:`genindex`
* :ref:`modindex`
