from setuptools import setup, find_packages
version = '1.0'

setup(
    name='pygeoiga',
    version=version,
    packages=find_packages(exclude=('test', 'docs')),
    include_package_data=True,
    install_requires=[
        'matplotlib >= 3.2.1',
        'numpy',
        'scipy',
        'ipython',
        'jupyter',
        'pyvista',
        'pytest',
        'PyQt5',
        'pygmsh==6.1.1',
        'gmsh',
        'meshio==4.0.3',
        'tqdm'

    ],
    url='https://github.com/danielsk78/master_thesis_IGA-Geothermal-Modeling',
    license='LGPL v3',
    author='Daniel Escallon',
    author_email='daniel.escallon@rwth-aachen.de',
    description='perform IGA and bezier extraction and manage plotting',
    keywords=['bezier', 'splines', 'nurbs', 'geology']
)
