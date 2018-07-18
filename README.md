# instanton_w_keras
This is designed to interpolate chemical potential energy surfaces in the
region around typical reaction paths.

Training data are hessians from quantum chemistry program outputs. A trained
network takes as input, the x,y,z coords of each atom in the molecule.

180630: Works fine for energies, accuracy isn't yet good enough for
gradients, once the keras team implements an L-BFGS optimizer things might
look better (or I'll do it myself).

The readin.f90 file MUST be compiled on your machine before executing neuralnetwork_sean_conv.py using:
f2py --fcompiler=gnu95 --f90flags=-fopenmp --opt=-O2 -c -m readin readin.f90 -llapack -lblas -lgomp
