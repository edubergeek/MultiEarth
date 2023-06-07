#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import sys
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv1DTranspose, Conv2DTranspose
from meETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, MultiEarthETL
from DLKeras import DLKeras


# %%


IS_JUPYTER = True
#TRANSFORM = 'transform'
AUTOTUNE = tf.data.AUTOTUNE
PROGRESS_INTERVAL = 10


# %%


if IS_JUPYTER:
  sys.argv.append('--model')
  sys.argv.append('s2vunet')
  sys.argv.append('--version')
  sys.argv.append('6')
  sys.argv.append('--trial')
  sys.argv.append('1')
  sys.argv.append('--epochs')
  sys.argv.append('100')
  sys.argv.append('--patience')
  sys.argv.append('5')
  sys.argv.append('--threshold')
  sys.argv.append('1e-6')
  sys.argv.append('--batch_size')
  sys.argv.append('128')
  sys.argv.append('--lr')
  sys.argv.append('1e-5')
  sys.argv.append('--arch')
  sys.argv.append('AE')
  sys.argv.append('--epsilon')
  sys.argv.append('10.0')
  sys.argv.append('--metric')
  sys.argv.append('mse')
  sys.argv.append('--loss')
  sys.argv.append('mse')
  sys.argv.append('--datadir')
  sys.argv.append('/me2023/tfr')
#  sys.argv.append('--trainpat')
#  sys.argv.append('valid_*[0-8].tfr')
#  sys.argv.append('--validpat')
#  sys.argv.append('valid_*9.tfr')   
  sys.argv.append('--xtransform')
  sys.argv.append('YXZ,32,32,2')
  sys.argv.append('--ytransform')
  sys.argv.append('YXZ,32,32,3')
  sys.argv.append('--train')

  print(sys.argv)


# %%


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
parser.add_argument("--version", type=int, default=1, help="model version")
parser.add_argument("--round", type=int, default=0, help="model round")
parser.add_argument("--revision", type=int, default=0, help="model revision")
parser.add_argument("--trial", type=int, default=1, help="training trial number")
parser.add_argument("--epoch", type=int, default=0, help="model epoch")
parser.add_argument("--begin", type=int, default=0, help="start with epoch")
parser.add_argument("--patience", type=int, default=0, help="early stopping patience")
parser.add_argument("--threshold", type=float, default=1e-5, help="early stopping threshold")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--xtransform", default='-', help="X transform")
parser.add_argument("--ytransform", default='-', help="Y transform")
parser.add_argument("--arch", default="NR", help="model architecture")
parser.add_argument("--ensemble1", default='-', help="1st ensemble model,arch,version")
parser.add_argument("--etransform", default='-', help="post-ensemble transform")
parser.add_argument("--ensemble2", default='-', help="2nd ensemble model,arch,version")
parser.add_argument("--batch_size", type=int, default=8192, help="batch size")
parser.add_argument('--train', action='store_true')
parser.add_argument('--trainera', action='store_true')
parser.add_argument('--datadir', default="./tfr", help='data directory')
parser.add_argument('--trainpat', default="train*.tfr", help='training file glob pattern')
parser.add_argument('--validpat', default="valid*.tfr", help='validation file glob pattern')
parser.add_argument('--monitor', default="val_loss", help='metric for checkpoint monitor')
parser.add_argument('--metric', default="mse", help='metrics')
parser.add_argument('--loss', default="mse", help='loss function')


# %%


if IS_JUPYTER:
  args = parser.parse_args(sys.argv[3:])
else:
  args = parser.parse_args()

print(args)



# %%


parameter = {
  'batch_size': args.batch_size,
}

parameter['name'] = args.model
parameter['train'] = args.train
parameter['round'] = args.round
parameter['trial'] = args.trial
parameter['revision'] = args.revision
parameter['version'] = args.version
parameter['epoch'] = args.epoch
parameter['epochs'] = args.epochs
parameter['begin'] = args.begin
parameter['patience'] = args.patience
parameter['threshold'] = args.threshold
parameter['learning_rate'] = args.lr
parameter['batch_size'] = args.batch_size
parameter['epsilon'] = args.epsilon
parameter['arch'] = args.arch
parameter['monitor'] = args.monitor
parameter['metric'] = args.metric
parameter['loss'] = args.loss
parameter['xtransform'] = args.xtransform
parameter['ytransform'] = args.ytransform
print(parameter)

# %%


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  sar2vis = DLKeras(parameter)
  sar2vis.ParseTransform(args.xtransform, args.ytransform)

# %%


ds = sar2vis.DataSet(args.datadir, args.trainpat)
dsv = sar2vis.DataSet(args.datadir, args.validpat)


# %%
print(args.datadir)
print(args.trainpat)
print(args.validpat)
# #!ls ./tfr/valid_*[0-8].tfr

# %%
#ex = dsv.take(1)
#for e in ex:
#  print(e)
#  exit(0)

# %%


with strategy.scope():
  # Train the model
  #sar2vis.path = os.path.join(".", sar2vis.GetModelName())
  sar2vis.LoadModel(useEpoch=False)
  sar2vis.SetModelTrial(1)
  #autoencoder = nm.model.get_layer('input_output')
  #autoencoder.trainable = False
  #z = nm.model.get_layer('z')
  #z.trainable = False
  sar2vis.model.summary()
  #sar2vis.SetMonitor(hyperParam['monitor'])


# %%


if args.train:
  print("Training")
  sar2vis.Train(ds, dsv)
  print("Done")
  path = sar2vis.path


# %%


cmd = "ls -lt %s" %('model')
os.system(cmd)


# %%
