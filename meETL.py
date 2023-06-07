#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from datetime import datetime
import os
import json
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from enum import IntEnum

from MultiEarth import Sentinel1, Sentinel2, MultiEarth
from ETL import ETL, TrainingSet
from ETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature


# %%
class MultiEarthETL(ETL):
  def __init__(self, root_dir, output_dir, shard_size = 1000, valid_split=0.01, byEra=False):
    train_split = 1.0 - valid_split
    super().__init__(root_dir, output_dir, shard_size, train_split, valid_split)
    dt = datetime.now()
    self.manifest = None
    self.me = None
    self.byEra = byEra
    self.trainSplit = 1.0 - valid_split
    self.validSplit = valid_split
    self.writer = None
    self.counter = [0,0,0,0]
    self.examples = [0,0,0,0]
    self.mask = [np.ones((1)),np.ones((1)),np.ones((1)),np.ones((1))]
    #self.OpenDatasets()

  def SetMultiEarth(self, me=None):
    self.me = me
    
  def Writer(self):
    if self.writer is None:
      self.writer = [ None,
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TRAIN, 0)),
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.VALID, 0)),
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TEST, 0)),
      ]
    return self.writer

  def OpenDatasets(self, byEra=False):
    # note that validation eras are > training examples and train begins with era 1
    self.eranum = 1
    if byEra:
      self.SetPartitionTag("era%d" % (self.eranum))
    self.Writer()

  def Split(self, mode, examples=None):
    if examples is None:
      examples = self.trainingExamples   
    self.mask[mode] = np.ones((examples), dtype=np.int8)
    
    nValid = int(self.validSplit * examples)
    filtered = np.random.choice(examples, nValid, replace=False)
    self.mask[mode][filtered] = 0
    self.examples[TrainingSet.TRAIN] += examples - nValid
    self.examples[TrainingSet.VALID] += nValid
    self.nExamples = examples

  def Load(self, mode, manifest, reload=False):
    if self.me is None:
      return

    # apply train valid test split
    # after this, check self.mask[mode] whether an index is in train (True) or valid (False)
    if not reload:
      self.manifest = np.loadtxt(manifest, delimiter=",", skiprows=1, dtype=int)
      self.trainingExamples = self.manifest.shape[0]
      self.Split(mode, self.trainingExamples)
         
    self.example = None
    x_data = np.arange(self.nExamples)
    y_data = np.copy(x_data)

    if mode == TrainingSet.TRAIN:
      self.trainIdx = x_data
    elif mode == TrainingSet.VALID:
      self.validIdx = x_data
    else:
      self.testIdx = x_data
    # Change to function that returns nExamples for current mode
    print("Loaded %d examples" %(self.nExamples))

  def Example(self, m):
    if self.era is None:
      return self.X[m], self.Y[m], self.id[m].as_py(), 0
    else:
      return self.X[m], self.Y[m], self.id[m].as_py(), self.era[m]

  def TrainingExample(self, sent1, sent2, idxs2):
    # Crop x and y images
    sent1.Crop(cropsize=32)
    sent2[idxs2].Crop(cropsize=32)
    # empty list for the selected examples
    example = []
    # for each crop see if it is selected and make an example of it
    for c in range(sent2[idxs2].GetCrops()):
      img2 = sent2[idxs2].GetCrop(crop=c)
      frq, bin_edges = np.histogram(img2)
      if bin_edges[-1] < 0.45:
        img1 = sent1.GetCrop(crop=c)
        x = img1.flatten()
        y = img2.flatten()
        feature = { 'x': _floatvector_feature(x), 'y': _floatvector_feature(y) }
        # Create an example protocol buffer
        example.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example, len(example)

  def SampleSet(self, m):
    S1 = self.example[m]
    S2 = self.vh[S1]
    vhPath = "%s/sent1/%s" % (self.rootDir, S1)
    vvPath = vhPath.replace("_VH_", "_VV_")
    s1 = Sentinel1(vhPath, vvPath)
    nS2 = int(len(S2)/3)
    sample = []
    for s in range(nS2):
      r = "%s/sent2/%s" % (self.rootDir, S2[2*nS2+s])
      g = "%s/sent2/%s" % (self.rootDir, S2[1*nS2+s])
      b = "%s/sent2/%s" % (self.rootDir, S2[s])
      s2 = Sentinel2(r, g, b)
      sample.append(s2)
    return s1, sample, nS2
        
  def SaveDataset(self, dataset, mode, byEra=False):
    cursor = mode
    #if byEra:
    #  self.SetPartitionTag("era%d" % (self.eranum))
    # Make sure we have initialized a writer
    
    for m in range(self.nExamples):
      # Take an example if selected for the current mode
      if mode == TrainingSet.TEST or self.mask[dataset][m] == (mode == TrainingSet.TRAIN):
        # get indices for this selected training example
        idxS1 = self.manifest[m][0]
        idxS2 = self.manifest[m][1]
        
        # get the example objects
        sent1, sent2, nSent2 = self.me.SampleSet(idxS1)
        example, examples = self.TrainingExample(sent1, sent2, idxS2)
        
        # if writing TFRecord per era check for a new era
        #if byEra and eranum != self.eranum:
        #  # if a new era then close the current era
        #  self.writer[cursor].close()
        #  # init the new era
        #  self.eranum = eranum
        #  self.SetPartitionTag("era%d" % (self.eranum))
        #  # todo save counter for each partition and cursor/mode
        #  self.counter[cursor] = 0
        #  self.writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, self.counter[cursor]))
  
        # write one (usually) or a set of correlated examples to the TFRecord file via the writer object
        for e in range(examples):
          # Check for reaching shard partition size and if so, close the shard and start a new one
          # for multiple examples should we really do this?
          if self.counter[cursor] % self.shardSize == 0:
            if self.counter[cursor] > 0:
              self.writer[cursor].close()
              self.writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, self.counter[cursor]))
          self.writer[cursor].write(example[e].SerializeToString())
          self.counter[cursor] += 1

  def CloseDatasets(self):
    if self.counter[TrainingSet.TRAIN] > 0:
      self.writer[TrainingSet.TRAIN].close()
    if self.counter[TrainingSet.VALID] > 0:
      self.writer[TrainingSet.VALID].close()
    if self.counter[TrainingSet.TEST] > 0:
      self.writer[TrainingSet.TEST].close()


def main():
  etl = MultiEarthETL(root_dir='/me2022', output_dir='/me2023/tfr', shard_size=4192, valid_split=0.1)
  me = MultiEarth()

  etl.SetMultiEarth(me)
  etl.OpenDatasets()

  #me.MakeManifest()

  etl.Load(TrainingSet.TRAIN, 'selected.csv')

  etl.SaveDataset(TrainingSet.TRAIN, TrainingSet.TRAIN)
  etl.SaveDataset(TrainingSet.VALID, TrainingSet.VALID)
  #etl.SaveDataset(TrainingSet.TEST, TrainingSet.TEST)

  etl.CloseDatasets()

if __name__ == '__main__':
  main()

