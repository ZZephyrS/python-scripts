# python-useful

Collection of some useful scripts I created for analyze model output (MITgcm and Lagrangian particle-tracking models) and data (from hydrographic sections, underway data, LADCP data and profiling core and biogeochemical Argo floats). 

Few useful libraries that I use in the scripts:

1) Gibbs-SeaWater (GSW) Oceanographic Toolbox that contains the TEOS-10 subroutines for evaluating the thermodynamic properties of pure water (http://www.teos-10.org/software.htm)

2) Very cool colormaps with cmocean (https://matplotlib.org/cmocean/)

3) Cmocean is found also in Palettable (https://jiffyclub.github.io/palettable/), which contains different and very cool colormaps (I really like colorbrewer, from http://colorbrewer2.org).

I usually install all the libraries using Anaconda: 

e.g. Palettable:  conda install -c conda-forge palettable 
