# pyGeoIGA
### Making sense of Isogeometric Analysis for geothermal applications: Parametric geomodeling (NURBS) for fast model construction, simulation and adaptation


Table of Contents
--------
* [Introduction](README.md#introduction)
* [Features](README.md#features)
* [Requirements](README.md#requirements)
* [Installation](README.md#installation)
* [Project Development](README.md#project-development)

Introduction
--------

Isogeometric analysis (IGA) is a technique that uses the power of the finite element method
(FEM) to numerically solve differential equations without the need of creating a discretization
in the space of the geometric object (i.e., a mesh). Instead, it uses computer-aided design
(CAD) tools, specifically Non-Uniform Rational B-splines (NURBS), to accurately represent
any geometry and perform analysis during design. This method has not been used in a Geoscientic
context, where the geometry of subsurface structures greatly influences the solution
of a simulation. This thesis introduces the isogeometric analysis technique to the Geoscientific 
community providing a Python package (https://github.com/danielsk78/pygeoiga) with
a simple but clear computer implementation. It differs from other implementations for dealing
with multipatch structures, focusing on geological modelling with multiple subdomains.
A series of numerical examples are presented to show the use of the technique for solving the
two-dimensional heat conduction problem compared to the traditional FEM. Results show
that IGA requires fewer degrees of freedom (DoF) for convergence of the solution. Additionally, 
the Bézier extractor operator is introduced, providing an element structure that can
incorporate IGA into existing finite element codes.

Features
--------

* [Creation of NURBS](tutorials/01_Create%20single%20patch%20NURBS%20and%20plot%20-%20Matplotlib%20and%20pyvista.ipynb): 
Create any curves and surfaces using NURBS
* [Creation of Multiple patches](tutorials/02_Create%20multipatch%20NURBS%20and%20plot.ipynb): Create multiple patches of NURBS
* [Refinement](tutorials/04_Multipatch%20global%20refinement.ipynb): Knot insertion and degree elevation for the refinement of patches
* [Topology](tutorials/08_Pre-processing%20workflow.ipynb): Define topological relations between patches
* [Assemble of stiffness matrix](tutorials/09_Assemble%20stiffness%20matrix%20and%20solve.ipynb): Routines for assembling stiffnes matrix and solving
* [Post-processing](tutorials/10_Complete%20workflow%20with%20salt%20dome%20model.ipynb): Plotting of solutions
* [Exports to FEniCS](tutorials/13_Export%20to%20FEniCS.ipynb) and [MOOSE](tutorials/14_Export%20to%20MOOSE.ipynb)
* [Bézier extraction operator](tutorials/15_Bezier_extraction_workflow.ipynb): Complete workflow for using the bezier extractor operator

Requirements
--------
    matplotlib >= 3.2.1
    numpy
    scipy
    ipython
    jupyter
    pyvista
    pytest
    PyQt5
    pygmsh==6.1.1
    gmsh
    meshio==4.0.3

(Optionals)

[igakit](https://bitbucket.org/dalcinl/igakit/src/master/): Fullly compatible with the nurbs this library produces. 
And the degree elevation routine depends completelly of this
```
pip install https://bitbucket.org/dalcinl/igakit/get/master.tar.gz
```  
[FEniCS](https://fenicsproject.org/): Finite element framework. Usefull for comparison and control of solutions. Follow the [instructions](https://fenicsproject.org/download/) for the installation

Installation
--------
For a local installation do:
```
pip install -e .
```

Project Development
-------------------

Author: Daniel Escallón




