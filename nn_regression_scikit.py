from __future__ import print_function
from __future__ import division
configfile = 'f_sources/readin.py'
import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.expanduser(configfile)))

import readin as rdin
import numpy as np
import scipy as sp
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, BatchNormalization, ActivityRegularization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras import optimizers, activations, initializers, regularizers, constraints
from levers_and_switches import inp_reader,fortran_float,inp_converter
from keras.engine.topology import Layer
from keras.wrappers.scikit_learn import KerasRegressor
import theano
theano.config.exception_verbosity='high'
import theano.tensor as T
import theano.d3viz as d3v
from theano import pp
from keras.utils import plot_model

if not os.path.exists('input_configs'):
    os.makedirs('input_configs')
inpfile=inp_reader('inp.dat')
inpfile.readlines(['!'])
configuration = inpfile.line_set_split
configuration = inp_converter(configuration)

#READING THE FORTRAN CONFIG FILE
[lambda_, alpha_] = configuration[0]
maxiter = int(configuration[1][0])
[pinv_tol_f, pinv_tol_b] = configuration[2]
PCA_TF = False
if(configuration[3] == 'T'):
  PCA_TF = True
coord_sys = int(configuration[4][0])
N_extracoords = int(configuration[5][0])
lbfgs_rate = configuration[6]
res_weights = configuration[11]
layer_extent = map(int,configuration[12])
pinv_type = int(configuration[13][0])

grads_hessians = False # True works only with model3.
coords_interatomic = False
coords_inverse = False
if(coord_sys == 1):
  coords_interatomic = True
elif(coord_sys == 2):
  coords_inverse = True
else:
  print('coord system must be either 1 or 2 (0 and 3 only work in the Fortran version).')
  quit()

[ncoord, ntrain, ntest, nat, dim_renorm] = rdin.pes_init(
#  path = 'data/HNCOH/training_HNCOH/',
#  pathtest = 'data/HNCOH/test_HNCOH/',
  path = 'data/Methanol+H/training/',
  pathtest = 'data/Methanol+H/test/',
  coords_interatomic = coords_interatomic,
  coords_inverse = coords_inverse,
  dimension_reduce = PCA_TF,
  remove_n_dims = 0,
  extra_red_coords = N_extracoords)

[refcoords, refene, refgrad, refhess, testcoords, testene, testgrad, testhess, xcoords_store, xgradient_store, xhessian_store,
xfreqs_store, xvectors_store, refmass, align_modes_all, align_refcoords, align_modes, DMAT, DMAT_PINV, DMAT_PINV2, projection, KMAT,
KMAT_Q, DM2, pca_eigvectr, mu, variance, radii_omit] = rdin.pes_read(
  ncoord_ = ncoord,
  npoint = ntrain,
  ntest = ntest,
  nat = nat,
  remove_n_dims = 0,
  dim_renorm = dim_renorm,
  coords_interatomic = coords_interatomic,
  coords_inverse = coords_inverse,
  dimension_reduce = PCA_TF,
  pinv_tol = pinv_tol_f,
  pinv_tol_back = pinv_tol_b,
  tol_type = pinv_type)

(x_train, y_train), (x_test, y_test) = (refcoords, refene), (testcoords, testene)
ytrainmean = y_train.mean()
ytrainstdv = y_train.std()
xtrainmean = x_train.mean(axis=1)
xtrainstdv = x_train.std(axis=1)
x_train = x_train.T#reshape(ntrain, ncoord)
x_test = x_test.T#reshape(ntest, ncoord)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')
grads_hessians = 1
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
if grads_hessians == 1 or grads_hessians == 2:
  y_train_gh = []
  y_test_gh = []
  iref = 1
  while iref <= ntrain:
    y_train_gh.append(np.append(np.insert(refgrad[:,iref-1],0,y_train[iref-1]),refhess[:,:,iref-1][np.triu_indices(ncoord)]))
    iref += 1
  y_train_gh = np.asarray(y_train_gh)
  itst = 1
  while itst <= ntest:
    y_test_gh.append(np.append(np.insert(testgrad[:,itst-1],0,y_test[itst-1]),testhess[:,:,itst-1][np.triu_indices(ncoord)]))
    itst += 1
  y_test_gh = np.asarray(y_test_gh)
#  y_train_gh = y_train_gh.astype('float32')
#  y_test_gh = y_test_gh.astype('float32')
  print('y_train_gh shape:', y_train_gh.shape)

num_classes = 1
mlp = MLPRegressor(hidden_layer_sizes = (layer_extent[0], layer_extent[1]), 
                  max_iter = maxiter,
                  activation = 'tanh',
                  solver = 'lbfgs',
                  learning_rate_init = lbfgs_rate,
                  verbose = True,
                  tol = 1e-16,
                  validation_fraction = 0.0,
                  abs_loss = False,
                  target_grad_hess = grads_hessians
                  )
if grads_hessians == 1 or grads_hessians == 2:
  mlp.fit(x_train, y_train_gh[:,:ncoord+1])
else:
  mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
print('Energies result\n', predictions)
print('Energies test\n', y_test)
print('Prediction, Target, Abs error, ave abs error, training std dev:\n', np.asarray(map(np.abs,(predictions - y_test))).mean(), 2626*np.asarray(map(np.abs,(predictions - y_test))).mean(), 'kJ/mol', 27.21*np.asarray(map(np.abs,(predictions - y_test))).mean(), 'eV', ytrainstdv)
