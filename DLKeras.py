import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv1DTranspose, Conv2DTranspose

import matplotlib
from matplotlib import pyplot as plt
from IPython.display import clear_output

TRANSFORM = [[],[]]

# Track the best epoch during training
class BestEpoch(Callback):
  def __init__(self, metric='val_loss', mode='min'):
    super().__init__()
    self.metric = metric
    self.mode = mode

  def on_train_begin(self, logs={}):
    self.bestEpoch = 0
    if self.mode == 'min':
      self.bestLoss = 1e8
    else:
      self.bestLoss = -1e8

  def on_epoch_end(self, epoch, logs={}):
    valLoss = logs.get(self.metric)
    if self.mode == 'min' and valLoss < self.bestLoss:
      self.bestLoss = valLoss
      self.bestEpoch = epoch+1
    elif valLoss > self.bestLoss:
      self.bestLoss = valLoss
      self.bestEpoch = epoch+1

  def get_best_epoch(self):
    return self.bestEpoch

# Plot the training vs validation loss
class PlotLoss(Callback):
  def __init__(self, metric='val_loss'):
    super().__init__()
    self.metric = metric
    matplotlib.interactive(True)

  def __del__(self):
    plt.close('all')

  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.loss_y = []
    self.metric_y = []
    self.logs = []
    self.fig = plt.figure()

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    self.loss_y.append(logs.get('loss'))
    self.metric_y.append(logs.get(self.metric))
    self.i += 1

    #clear_output(wait=True)
    plt.clf()
    plt.plot(self.x, self.loss_y, label="loss")
    plt.plot(self.x, self.metric_y, label=self.metric)
    plt.legend()
    plt.show(block=False);
    plt.draw();
    plt.pause(0.05)
    print(logs)

def decode_tfr(record_bytes):
    schema =  {
      "x":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
      "y":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
    }
    example = tf.io.parse_single_example(record_bytes, schema)
    return example['x'], example['y']

def reshape_YXZ(features, targets):
  for t in TRANSFORM[0]:
    if t['name'] == 'YXZ':
      yDim = int(t['arg1'])
      xDim = int(t['arg2'])
      zDim = int(t['arg3'])
      features = tf.reshape(features, [yDim, xDim, zDim])
      #features = tf.transpose(features)
  for t in TRANSFORM[1]:
    if t['name'] == 'YXZ':
      yDim = int(t['arg1'])
      xDim = int(t['arg2'])
      zDim = int(t['arg3'])
      targets = tf.reshape(targets, [yDim, xDim, zDim])
  return features, targets

def remap_autoencoder(features, targets):
    return features, (features, targets)


#param dict values:
#   'arch' in ['AE','MC', 'LR'] # AE = AutoEncoder, MC = multi-classifier, LR = Logistic Regression
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
class DLKeras():
  def __init__(self, param ):
    self.param = param
    self.name = param['name']
    self.version = param["version"]
    self.trial = param["trial"]
    self.batchSize = param["batch_size"]
    if 'epochs' in param.keys():
      self.epochs = param["epochs"]
    else:
      self.epochs = 0        
    if 'patience' in param.keys():
      self.stopPatience = param["patience"]
    else:
      self.stopPatience = 0
    if 'threshold' in param.keys():
      self.stopThreshold = param["threshold"]
    else:
      self.stopThreshold = 0.0
    if 'learning_rate' in param.keys():
      self.learningRate = param["learning_rate"]
    else:
      self.learningRate = 0.0
    if 'begin' in param.keys():
      self.begin = param['begin']
    else:
      self.begin = 0
    if 'path' in param.keys():
      self.path = param['path']
    else:
      self.path = './model'
    if 'epoch' in param.keys():
      self.epoch = param["epoch"]
    else:
      self.epoch = 0
    if 'monitor' in param.keys():
      self.monitor = param['monitor']
    else:
      self.monitor = 'val_loss'
    if 'metric' in param.keys():
      self.metrics = [param['metric']]
    else:
      self.metrics = ['mse']
    if 'loss' in param.keys():
      self.loss = param['loss']
    else:
      self.loss = 'mse'
    if 'mode' in param.keys():
      self.mode = param['mode']
    else:
      if self.monitor == 'val_accuracy':
        self.mode = 'max'
      else:
        self.mode = 'min'
    self.saveBestOnly = True
    self.saveWeightsOnly = False
    self.optimizer = Adam(learning_rate=self.learningRate)
    self.step = 0   #Not currently used
    self.ylabels = None
    self.isTrained = False

  def GetModelFile(self, useEpoch=True):
    if useEpoch:
      return "%s/%s-v%dt%d-e%d" %(self.path, self.name, self.version, self.trial, self.epoch)
    else:
      return "%s/%s-v%dt%d" %(self.path, self.name, self.version, self.trial)

  def LoadModel(self, useEpoch=True):
    self.modelFile = self.GetModelFile(useEpoch)
    print("Loading ", self.modelFile)
    self.model = tf.keras.models.load_model(self.modelFile)
    if self.param['arch'] == 'MC':
      self.Compile(self.model, loss='sparse_categorical_crossentropy', metric=['accuracy'])
    elif self.param['arch'] == 'AE':
      self.Compile(self.model, loss=self.loss)
    else:
      self.Compile(self.model, loss=self.loss)

  def GetBestEpochFile(self):
    return "%s/%s-best.csv" %(self.path, self.name)

  def LoadBestEpochs(self, predict='-'):
    self.bestEpochFile = self.GetBestEpochFile()
    print('BestEpochFile = ', self.bestEpochFile)
    if os.path.exists(self.bestEpochFile):
      self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
    else:
      self.bestEpochFile = "%s/%s-best.csv" %(self.path, predict)
      if os.path.exists(self.bestEpochFile):
        self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
    print(self.best)

  def GetTrialModelFile(self, model, version, trial, epoch, predict):
    if predict == '-':
      return "%s/%s-v%dt%d-e%d" %(self.path, model, version, trial, epoch)
    else:
      return "%s/%s/%s-v%dt%d-e%d" %(self.path, predict, model, version, trial, epoch)

  def GetModelFullName(self):
    if self.step:
      return "%s-v%ds%d" %(self.name, self.version, self.step)
    return "%s-v%d" %(self.name, self.version)

  def GetModelName(self):
    return self.name

  def GetModelTrial(self):
    return self.trial

  def SetModelTrial(self, trial):
    self.trial = trial

  def GetList(self, f):
    return f.numpy().tolist()

  def SetEpochs(self, epochs):
    self.epochs = epochs

  def SetBatchSize(self, batch_size):
    self.batchSize = batch_size

  def SetModel(self, model):
    self.model = model

  def SetModelName(self, name):
    self.name = name

  def SetModelVersion(self, version):
    self.version = version

  def GetModelVersion(self):
    return self.version

  def SetModelTrial(self, trial):
    self.trial = trial

  def GetModelTrial(self):
    return self.trial

  def SetLearningRate(self, rate):
    self.learningRate = rate

  def SetMonitor(self, monitor):
    self.monitor = monitor
    if monitor == 'val_accuracy':
      self.mode = 'max'

  def SetLoss(self, loss):
    self.loss = loss

  def SetMetrics(self, metrics):
    self.metrics = metrics

  def SaveBestOnly(self):
    self.saveBestOnly = True

  def SaveEveryEpoch(self):
    self.saveBestOnly = False

  def SaveWeightsOnly(self):
    self.saveWeightsOnly = True

  def SaveFullModel(self):
    self.saveWeightsOnly = False

  def SetLearningRate(self, rate):
    self.learningRate = rate

  def SetOptimizer(self, optimizer):
    self.optimizer = optimizer(learning_rate=self.learningRate)

  def ParseTransform(self, xtransform, ytransform):
    if not xtransform == '-':
      trans = []
      filters = xtransform.split('|')
      for f in range(len(filters)):
        param = filters[f].split(',')
        if len(param) == 3:
          trans.append({
            'name': param[0],
            'arg1': param[1],
            'arg2': param[2]
            })
        if len(param) == 4:
          trans.append({
          'name': param[0],
            'arg1': param[1],
            'arg2': param[2],
            'arg3': param[3]
            })
      self.xTransform = trans
      TRANSFORM[0] = trans
    if not ytransform == '-':
      trans = []
      filters = ytransform.split('|')
      for f in range(len(filters)):
        param = filters[f].split(',')
        if len(param) == 3:
          trans.append({
            'name': param[0],
            'arg1': param[1],
            'arg2': param[2]
            })
        if len(param) == 4:
          trans.append({
          'name': param[0],
            'arg1': param[1],
            'arg2': param[2],
            'arg3': param[3]
            })
      self.yTransform = trans
      TRANSFORM[1] = trans

  def GetDataSet(self, filenames):
    at = tf.data.AUTOTUNE
    #TRANSFORM = [self.xTransform, self.yTransform]

    print(filenames)
    dataset = (
      tf.data.TFRecordDataset(filenames, num_parallel_reads=at)
      .map(decode_tfr, num_parallel_calls=at)
    )

    for t in TRANSFORM[0]:
      if t['name'] == 'YXZ':
        dataset = dataset.map(reshape_YXZ, num_parallel_calls=at)

    if self.param['arch'] == 'AE':
      dataset = dataset.map(remap_autoencoder, num_parallel_calls=at)

    dataset = dataset.batch(self.batchSize).prefetch(at).repeat(count=1)

    return dataset

  def DataSet(self, path, pattern):
    pattern_list = pattern.split()
    filenames = []
    for pat in pattern_list:
      filenames += tf.io.gfile.glob(os.path.join(path, pat))
    return self.GetDataSet(filenames)

  def ResetWeights(self):
    if self.isTrained:
      weights = []
      initializers = []
      for layer in self.model.layers:
        if isinstance(layer, (Dense, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose)):
          weights += [layer.kernel, layer.bias]
          initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, BatchNormalization):
          weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
          initializers += [layer.gamma_initializer,
                       layer.beta_initializer,
                       layer.moving_mean_initializer,
                       layer.moving_variance_initializer]
      for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))

    self.isTrained = False

 #def Compile(self):
 #  self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

  def Compile(self, model, loss = None, loss_weight=None, ylabels=None):
    # compile the model
    if self.param['arch'] == 'AE':
      if ylabels is None:
        ylabels = ['xhat', 'yhat']
      metric = [loss]
      losses = {
        'xhat': loss,
        'yhat': loss,
      }
      lossWeights = {'xhat': 1.0, 'yhat': self.param['epsilon']}
      model.compile(optimizer=self.optimizer, loss=losses, loss_weights=lossWeights, metrics=metric)
    else:
      model.compile(optimizer=self.optimizer, loss=[self.loss], metrics=self.metrics)
    self.model = model
    return self.model

  def Train(self, ds, dsv):
    # Set the model file name
    filepath="%s/%st%d-e{epoch:d}" %(self.path, self.GetModelFullName(), self.GetModelTrial())
    # default checkpoint settings
    checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1, save_best_only=self.saveBestOnly, save_weights_only=self.saveWeightsOnly, mode=self.mode)
    # plot loss after each epoch
    plotloss = PlotLoss(metric=self.monitor)
    bestepoch = BestEpoch(metric=self.monitor, mode=self.mode)
    if self.stopPatience > 0:
      earlystop = EarlyStopping(monitor=self.monitor, mode=self.mode, patience=self.stopPatience, min_delta=self.stopThreshold)
      self.callbacks = [checkpoint, plotloss, bestepoch, earlystop]
    else:
      self.callbacks = [checkpoint, plotloss, bestepoch]
    self.bestEpoch = 0

    # TODO train lgbm and xgboost models
    #if arch == 'XGB':
    #  model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
    # model.Train()
    # elif arch == 'LGB':
    #  model.Train()
    if not (self.param['arch'] == 'AE' or self.param['arch'] == 'MC' or self.param['arch'] == 'LR' or self.param['arch'] == 'UN'):
      print("Unsupported arch=%s in Train" %(self.param['arch']))
      return 0

    self.model.fit(ds, validation_data=dsv, initial_epoch=self.begin, epochs=self.epochs, batch_size=self.batchSize, callbacks=self.callbacks, verbose=1, shuffle=1)

    self.isTrained = True
    self.bestEpoch = bestepoch.get_best_epoch()
    return self.bestEpoch

  def SaveModel(self, useEpoch=True):
    self.model.save(self.GetModelFile(useEpoch))

  #def Predict(self, ds, transform):
  def Predict(self, x):
    # Generate predictions
    batchSize = min(len(x),self.batchSize)
    if self.param['arch'] == 'AE':
      xPred, yPred = self.model.predict(x, batchSize)
    else:
      yPred = self.model.predict(x, batchSize)
      xPred = None
    return [yPred, xPred]



