# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, concatenate, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Conv2DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from DLKeras import DLKeras

VERSION = 6

# -

class S2VUNet(DLKeras):
  def __init__(self, hparam):
    super().__init__(hparam)

    # Construct the model
    model = self.BuildModel(hparam)

    self.SetBatchSize(hparam['batch_size'])
    self.SetLearningRate(hparam['learning_rate'])

    self.SetModelName("s2vunet")
    self.SetModelVersion(1)
    self.SetModelTrial(1)

    self.Compile(model)
    #dt = datetime.now()
    #self.SetEpochs(50)

  def AutoEncoder(self, inputs, dim=32, layers=3, expand=2, kernel=3, activation='relu', batchNorm=True, needDense=False):
    # Construct the model
    act = activation
    pad = 'same'
    pool = inputs
    skip_conn = []
    for l in range(layers):
      print('encoder', l, dim, 'filters')
      m = Conv2D(filters=dim, kernel_size=kernel, padding=pad)(pool)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      if not batchNorm and self.dropout[l] > 0.0:
        #m = SpatialDropout2D(self.dropout[l])(m)
        m = Dropout(self.dropout[l])(pool)
      print('encoder', l, dim, 'filters')
      m = Conv2D(filters=dim, kernel_size=kernel, padding=pad)(pool)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      if not batchNorm and self.dropout[l] > 0.0:
        #m = SpatialDropout2D(self.dropout[l])(m)
        m = Dropout(self.dropout[l])(pool)
      skip_conn.append(m)
      if l+1 < layers:
        pool = MaxPooling2D(pool_size=2)(m)
      dim *= expand
    # latent embedding information bottleneck
    #m = Conv2D(filters=8, kernel_size=kernel, padding=pad, activation=act, name="z")(m)
    encoder = m
    for l in range(layers):
      dim /= expand
      print('encoder', layers+l, dim, 'filters')
      t = Conv2DTranspose(filters=dim, kernel_size=kernel, strides=2, padding=pad)(m)
      #m = t
      # Skip connection goes here ... concatenate corresponding input layer from above
      if l < (layers - 1):
        #print('concatenate', layers-l-2)
        #m = concatenate([t, skip_conn[layers-l-2]], axis=3)
        m = t
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      print('encoder', layers+l, dim, 'filters')
      m = Conv2D(filters=dim, kernel_size=kernel, padding=pad)(m)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      if not batchNorm and self.dropout[l] > 0.0:
        #m = SpatialDropout2D(self.dropout[l])(m)
        m = Dropout(self.dropout[l])(m)

    # final shape is (self.yDim/2**layers, self.xDim/2**layers, dim)
    if needDense:
      m = Flatten()(m)
      m = Dense(self.yDim*self.xDim, activation=act, name='flat')(m)
      m = Reshape((self.yDim,self.xDim,1), name="matrix")(m)
    else:
      m = Conv2D(filters=2, kernel_size=kernel, padding=pad, activation='linear',  name="xhat")(m)

    return m, encoder, skip_conn

  def Decoder(self, inputs, skip_conn, dim=32, layers=3, expand=2, kernel=3, activation='relu', batchNorm=True, needDense=False):
    # Construct the model
    act = activation
    pad = 'same'
    m = inputs
    for l in range(layers):
      dim /= expand
      print('decoder', layers+l, dim, 'filters')
      t = Conv2DTranspose(filters=dim, kernel_size=kernel, strides=2, padding=pad)(m)
      # Skip connection goes here ... concatenate corresponding input layer from above
      if l < (layers - 1):
        print('concatenate', layers-l-2)
        m = concatenate([t, skip_conn[layers-l-2]], axis=3)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      print('decoder', layers+l, dim, 'filters')
      m = Conv2D(filters=dim, kernel_size=kernel, padding=pad)(m)
      if batchNorm:
        m = BatchNormalization()(m)
      m = activations.relu(m)
      if not batchNorm and self.dropout[l] > 0.0:
        #m = SpatialDropout2D(self.dropout[l])(m)
        m = Dropout(self.dropout[l])(m)

    # final shape is (self.yDim/2**layers, self.xDim/2**layers, dim)
    if needDense:
      m = Flatten()(m)
      m = Dense(self.yDim*self.xDim, activation=act, name='flat')(m)
      m = Reshape((self.yDim,self.xDim,1), name="matrix")(m)
    else:
      m = Conv2D(filters=3, kernel_size=kernel, padding=pad, activation='linear',  name="yhat")(m)

    return m

  def BuildModel(self, hp):
    #self.yDim = 104
    #self.xDim = 8
    self.yDim = 32
    self.xDim = 32
    self.zDim = 2

    hp_activation = hp['activation']
    hp_units = hp['units']
    hp_expansion = hp['expansion']
    hp_layers = hp['layers']
    hp_dropout = hp['dropout']
    hp_batchnorm = hp['batchnorm']
    hp_kernel = hp['kernel']
#    hp_finalUnits = hp['final_units']
#    hp_finalLayers = hp['final_layers']
#    hp_finalExpansion = hp['final_expansion']
#    hp_finalDropout = hp['final_dropout']
#    hp_finalActivation = hp['final_activation']


    # Construct the model
    # initialize the input shape and channel dimension
    inputShape = (self.yDim, self.xDim, self.zDim)
    print(inputShape)
    # construct both the "autoencoder" and "regression" sub-networks
    inputs = Input(shape=inputShape, name='sar')
    # initialize dropout for each layer
    self.dropout=np.full((hp_layers),hp_dropout)

    unet, encoder, skip = self.AutoEncoder(inputs, dim=hp_units, layers=hp_layers, expand=hp_expansion, kernel=hp_kernel, activation=hp_activation, batchNorm=hp_batchnorm)

    #self.dropout=np.full((hp_finalLayers),hp_finalDropout)
    #mRegression = self.RegressionModel(inputs, latents, units=hp_finalUnits, layers=hp_finalLayers, expand=hp_finalExpansion, activation=hp_finalActivation, batchNorm=hp_batchnorm)
    decoder = self.Decoder(encoder, skip, dim=hp_units*(hp_layers)*hp_expansion, layers=hp_layers, expand=hp_expansion, kernel=hp_kernel, activation=hp_activation, batchNorm=hp_batchnorm)

    self.model = Model(inputs=inputs, outputs=[unet, decoder], name=self.name)
    return self.model


# +
def main():
#param dict values:
#   'arch' in ['AE','MC', 'LR', 'UN'] # AE = AutoEncoder, MC = multi-classifier, LR = Logistic Regression, UN=UNet
#   'name'        model name
#   'path'        path where models are stored
#   'version'     version of the model architecture
#   'trial'       version of weights e.g. for hyperparameter search
#   'epoch'       Epoch to load
#   'patience'    early stopping patience for Keras
#   'threshold'   early stopping threshold
#   'epochs'      number of epochs to train
#   'begin'       epoch at which to start training; default is zero
#   'batch_size'  batch size for training and inference
#   'metric'      metric used for loss function in Compile; default mse
#   'monitor'     metric to monitor for validation loss; default val_loss
#   'mode'        Minimize or maximize objective function; default min
#   'transform'   string parsed into a sequence of transfoms to apply to the input TFRecords
#   'epsilon'   scaling for input vs output loss for autoencoder loss function

  if VERSION == 2:
    hyperParam = {
      'name': 's2vunet',
      'arch': 'UN',
      'loss': 'mse',
      'version': '1',
      'trial': '1',
      'activation': 'relu',
      'units': 128,
      'expansion': 2,
      'layers': 2,
      'dropout': 0.5,
      'patience': 3,
      'threshold': 1e-5,
      'batchnorm': False,
      'epochs': 50,
      'kernel': 3,
      'final_units': 64, # was 128
      'final_layers': 2,
      'final_expansion': 2,
      'final_dropout': 0.0,
      'final_activation': 'relu',
      'learning_rate': 3e-5,
      'batch_size': 256,
    }


# -

  if VERSION == 3:
    hyperParam = {
      'name': 's2vunet',
      'arch': 'UN',
      'loss': 'mse',
      'version': '1',
      'trial': '1',
      'activation': 'relu',
      'units': 64,
      'expansion': 2,
      'layers': 4,
      'dropout': 0.5,
      'patience': 3,
      'threshold': 1e-5,
      'batchnorm': False,
      'epochs': 50,
      'kernel': 3,
      'final_units': 64, # was 128
      'final_layers': 2,
      'final_expansion': 2,
      'final_dropout': 0.0,
      'final_activation': 'relu',
      'learning_rate': 3e-5,
      'batch_size': 256,
    }

  if VERSION == 4:
    hyperParam = {
      'name': 's2vunet',
      'arch': 'UN',
      'loss': 'mse',
      'version': '1',
      'trial': '1',
      'activation': 'relu',
      'units': 256,
      'expansion': 2,
      'layers': 2,
      'dropout': 0.2,
      'batchnorm': False,
      'epochs': 100,
      'kernel': 3,
      'patience': 6,
      'threshold': 1e-6,
      'learning_rate': 3e-5,
      'batch_size': 256,
    }


  if VERSION == 5:
    hyperParam = {
      'name': 's2vunet',
      'arch': 'UN',
      'loss': 'mse',
      'version': '1',
      'trial': '1',
      'activation': 'relu',
      'units': 128,
      'expansion': 2,
      'layers': 2,
      'dropout': 0.5,
      'patience': 3,
      'threshold': 1e-5,
      'batchnorm': False,
      'epochs': 50,
      'kernel': 3,
      'final_units': 64, # was 128
      'final_layers': 2,
      'final_expansion': 2,
      'final_dropout': 0.0,
      'final_activation': 'relu',
      'learning_rate': 3e-5,
      'batch_size': 256,
    }


 if VERSION == 6:
    hyperParam = {
      'name': 's2vunet',
      'arch': 'UN',
      'loss': 'mse',
      'version': '1',
      'trial': '1',
      'activation': 'relu',
      'units': 128,
      'expansion': 2,
      'layers': 2,
      'dropout': 0.0,
      'patience': 3,
      'threshold': 1e-5,
      'batchnorm': True,
      'epochs': 50,
      'kernel': 3,
      'final_units': 64, # was 128
      'final_layers': 2,
      'final_expansion': 2,
      'final_dropout': 0.0,
      'final_activation': 'relu',
      'learning_rate': 3e-5,
      'batch_size': 256,
    }


  s2v = S2VUNet(hyperParam)
  print(s2v.model.summary())
  s2v.SetModelVersion(VERSION)
  s2v.SaveModel(useEpoch=False)
# + endofcell="--"
# -
if __name__ == '__main__':
  main()
# --

