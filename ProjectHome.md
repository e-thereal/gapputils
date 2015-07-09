# Introduction #

This is the website of grapevine, the gapputils rapid application prototyping environment and visualization network.

## Overview ##

Each gapputils application consists of three basic components:
  * The application host
  * The application description
  * Applications modules

All gapputils applications are executed in a so called **application host**. This program provides a very basic and highly extendable GUI which can be modified at runtime. Modifications done to the GUI are stored in the **application description**. This is an XML file describing the GUI and the interaction between modules. **Application modules** are the building blocks of a gapputils application. Modules are written in C++ and stored as dynamically loadable libraries (dll, or so).

More information can be found at the official blog at http://gapputils.blogspot.com/.

### Related Projects ###

  * capputils: http://code.google.com/p/capputils/

## User Interface ##

Modules are shown as small boxes with inputs and outputs and a name. Inputs and outputs are labels horizontally. The module itself is labeled vertical. Properties are given in a sidepane according to the current selection. Clicking at the ground shows the parameters of the toplevel grouping module.

## Screenshots ##

![http://gapputils.googlecode.com/svn/wiki/mainwindow1.png](http://gapputils.googlecode.com/svn/wiki/mainwindow1.png)