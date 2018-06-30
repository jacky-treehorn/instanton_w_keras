# NOTES 180515: Added a boolean grads_hessians (default = True) which when true means you need to expand the targets (y_train, y_test etc) to 
# include grads and hessians (upper triangle only though) on the PES. Currently only an internal residual/loss function is possible.

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
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, BatchNormalization
from keras import backend as K
from keras import optimizers, activations, initializers, regularizers, constraints
from levers_and_switches import inp_reader,fortran_float,inp_converter
from keras.engine.topology import Layer
import theano
import theano.tensor as T
from theano import pp
from keras.utils import plot_model

class bias_only(Layer):

    def __init__(self, input_shape, bias_initializer='zeros', **kwargs):
        self.output_dim = input_shape
        self.bias_initializer = initializers.get(bias_initializer)
        super(bias_only, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim),
                                    initializer=self.bias_initializer,
                                    trainable=True)
        super(bias_only, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        output = x + self.bias
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class dout_din(Layer):

    def __init__(self, output_shape, input_layer, **kwargs):
        self.output_dim = output_shape
        self.input_layer = input_layer
        super(dout_din, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(dout_din, self).build(input_shape)  # Be sure to call this at the end

    def call(self, y):
        output = K.gradients(K.sum(y[:,0]), self.input_layer)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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
res_weights = configuration[11]
layer_extent = map(int,configuration[12])
pinv_type = int(configuration[13][0])

grads_hessians = True # True works only with model2.
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

batch_size = ntrain #Use all geometries in training set for gradient updates.

(x_train, y_train), (x_test, y_test) = (refcoords, refene), (testcoords, testene)
ytrainmean = y_train.mean()
ytrainstdv = y_train.std()

if grads_hessians:
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
  gradmean = y_train_gh[:,1:ncoord+1].mean(axis = 0)
  gradstdv = y_train_gh[:,1:ncoord+1].std(axis = 0)

##Declare model specific parameters here#C4H7
#conv_window = 11
#pool_size = 6
#num_classes = 1
#stride_len = 5
#filters = 11
#Declare model specific parameters here#Methanol + H
conv_window = 6
pool_size = 3
num_classes = 1
stride_len = 2
filters = 11

model = Sequential()
model.add(Conv1D(filters, conv_window, strides = stride_len, activation = 'tanh', use_bias = True, bias_initializer = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None), input_shape = (ncoord, 1)))
model.add(AveragePooling1D(pool_size = pool_size, strides = stride_len))
model.add(Flatten())
model.add(Dense(layer_extent[0], activation = 'tanh', use_bias = True, bias_initializer = keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None)))
model.add(Dense(layer_extent[1], activation = 'tanh', use_bias = True, bias_initializer = keras.initializers.RandomUniform(minval = -1, maxval = 1, seed = None)))
model.add(Dense(num_classes, activation = 'linear', use_bias = True, bias_initializer = keras.initializers.Constant(value=y_train.mean())))

input_data = Input(shape=(ncoord, ), batch_shape = (batch_size, ncoord))
x = Dense(layer_extent[0], activation = 'tanh', use_bias = True, bias_initializer = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None))(input_data)
x = Dense(layer_extent[1], activation = 'tanh', use_bias = True, bias_initializer = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None))(x)
x = Dense(num_classes, activation = 'linear', use_bias = True, bias_initializer = keras.initializers.Constant(value = ytrainmean))(x)
energy = x
model2 = Model(inputs = input_data, outputs = energy)
energy_grad = dout_din(num_classes*ncoord, input_data)(energy)
model3 = Model(inputs = input_data, outputs = [energy, energy_grad])
print('Model 1 summary')
model.summary()
print('Model 2 summary')
if(not grads_hessians):
  model2.summary()
else:
  model3.summary()

cont = raw_input('Which model? 1 or 2 (any other button will quit)\n')
if(cont == '1'):
  if(grads_hessians):
    print('model1 not implemented with PES gradients and Hessians yet.')
    quit()
  x_train = x_train.reshape(ntrain, ncoord, 1)
  x_test = x_test.reshape(ntest, ncoord, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  y_train = y_train.reshape(ntrain, 1)
  y_test = y_test.reshape(ntest, 1)
  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')
  print('x_train shape:', x_train.shape)
  print('y_train shape:', y_train.shape)
  if grads_hessians:
    y_train_gh = y_train_gh.astype('float32')
    y_test_gh = y_test_gh.astype('float32')
    print('y_train_gh shape:', y_train_gh.shape)

  model.compile(loss = keras.losses.mean_absolute_error, optimizer = 'Adadelta')

  model.fit(x_train, y_train,
            batch_size = batch_size,
            epochs = maxiter,
            verbose = 1,
            validation_data = (x_test, y_test),
            shuffle = True
  )
  results = model.predict(x_test, verbose = 0)
  model.summary()
  print('Prediction, Target, Abs error, ave abs error:', np.asarray(map(np.abs,results - y_test)).mean(),2626*np.asarray(map(np.abs,results - y_test)).mean(),'kJ/mol',27.21*np.asarray(map(np.abs,results - y_test)).mean(),'eV')
elif(cont == '2'):
  x_train = x_train.reshape(ntrain, ncoord)
  x_test = x_test.reshape(ntest, ncoord)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')
  print('x_train shape:', x_train.shape)
  print('y_train shape:', y_train.shape)
  if grads_hessians:
    y_train_gh = y_train_gh.astype('float32')
    y_test_gh = y_test_gh.astype('float32')
    print('y_train_gh shape:', y_train_gh.shape)


  if grads_hessians:
    model3.compile(loss = keras.losses.mean_absolute_error,
                optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=0.0, decay=0.00001),
                loss_weights = [1.0, 1.0])
    model3.fit(x_train, [y_train, y_train_gh[:,1:ncoord+1]],
          batch_size = batch_size,
          epochs = maxiter,
          verbose = 2,
          validation_data = (x_test, [y_test, y_test_gh[:,1:ncoord+1]]),
          shuffle = True
          )
    results = model3.predict(x_test, verbose = 0)
  else:
    model2.compile(loss = keras.losses.mean_absolute_error,
                optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=0.0, decay=0.00001))
    model2.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = maxiter,
              verbose = 2,
              validation_data = (x_test, y_test),
              shuffle = True
              )
    results = model2.predict(x_test, verbose = 0)
  print('Energies result-test')
  print('Prediction, Target, Abs error, ave abs error:', np.asarray(map(np.abs,results[0] - y_test)).mean(), 2626*np.asarray(map(np.abs,results[0] - y_test)).mean(), 'kJ/mol', 27.21*np.asarray(map(np.abs,results[0] - y_test)).mean(), 'eV')
  if grads_hessians:
    i = 1
    while i <= ntest:
      dist = np.linalg.norm(y_test_gh[i-1,1:1+ncoord]-results[1][i-1,:])
      norm_targ = np.linalg.norm(y_test_gh[i-1,1:1+ncoord])
      print(i, dist, str(dist*100.0/norm_targ)+' %')
      i += 1
    print('Gradients: Prediction, Target, Abs error, ave abs error:', np.asarray(map(np.abs,results[1][:,:] - y_test_gh[:,1:ncoord+1])).mean())
else:
  quit()
