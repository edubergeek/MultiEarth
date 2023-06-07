#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from datetime import datetime
import json
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from enum import IntEnum


# %%


class Sentinel1():
  def __init__(self, vh_path, vv_path):
    super().__init__()
    self.vhPath = vh_path
    self.vvPath = vv_path
    self.isLoaded = False
    self.isStacked = False
    self.nCrop = 0
    self.Load()
    
  def Load(self):
    self.imvh = skio.imread(self.vhPath, plugin="tifffile")
    self.imvv = skio.imread(self.vvPath, plugin="tifffile")
    self.isLoaded = True
                        
  def as_numpy(self):
    if self.isLoaded:
      if not self.isStacked:
        self.s1 = np.dstack((self.imvh,self.imvv))
      return self.s1
    else:
        raise Exception("Sentinel1 as_numpy without prior Load")
    
  def as_feature(self):
    if not self.isStacked:
      self.as_numpy()
    a=self.s1.flatten(order='C')
    return _floatvector_feature(a.tolist())
#    return _dtype_feature(self.s1)

  def Normalize(self):
    if self.isLoaded:
      img = self.as_numpy()
      img -= np.mean(img)
      img_std = np.std(img)
      img += img_std
      img /= img_std * 4.0
      img[np.isnan(img)] = 0.0
      img = np.clip(img, 0, 1)
      return img
    else:
        raise Exception("Sentinel1 Normalize without prior Load")

  def Crop(self, cropsize=32):
    self.image = self.Normalize()    
    # assume square baseline image which is dangerous
    self.size = self.image.shape[0]
    self.cropSize = cropsize
    self.cropsPerRow = np.floor(self.size / cropsize)
    self.nCrop = int(self.cropsPerRow * self.cropsPerRow)

  def GetCrops(self):
    return self.nCrop
        
  def GetCrop(self, crop=0):
    r = int(crop / self.cropsPerRow)
    c = int(crop % self.cropsPerRow)
    r = r * self.cropSize
    c = c * self.cropSize
    return self.image[r:r+self.cropSize,c:c+self.cropSize,:]
        
  def GetCropStats(self, crop=0):
    r = int(crop / self.cropsPerRow)
    c = int(crop % self.cropsPerRow)
    r = r * self.cropSize
    c = c * self.cropSize
    img = self.image[r:r+self.cropSize,c:c+self.cropSize,:]
    return img.mean(), img.std(), img.max(), img.min()
        
  def Plot(self, channel=0):
    if not self.isStacked:
      self.as_numpy()
    if channel < 2:
      plt.imshow(self.s1[:,:,channel])
      plt.show

  def Statistics(self):
    if not self.isStacked:
      self.as_numpy()
    print('Shape: %d,%d,%d' % (self.s1.shape) )
    print('Mean: %f Std: %f' %(self.s1.mean(), self.s1.std()) )

  def Histogram(self, bins=10, channel=0):
    if not self.isStacked:
      self.as_numpy()
    if channel < 2:
      plt.hist(self.s1[channel], bins=bins)


# %%


class Sentinel2():
  def __init__(self, b4_path, b3_path, b2_path):
    super().__init__()
    self.b4Path = b4_path
    self.b3Path = b3_path
    self.b2Path = b2_path
    self.isLoaded = False
    self.isStacked = False
    self.Load()
    
  def Load(self):
    
    self.imb4 = skio.imread(self.b4Path, plugin="tifffile").astype(np.float32, order='C')
    self.imb3 = skio.imread(self.b3Path, plugin="tifffile").astype(np.float32, order='C')
    self.imb2 = skio.imread(self.b2Path, plugin="tifffile").astype(np.float32, order='C')
    self.isLoaded = True
   
  def as_numpy(self):
    if self.isLoaded:
      if not self.isStacked:
        self.s2 = np.dstack((self.imb4,self.imb3,self.imb2))
      return self.s2
    else:
        raise Exception("Sentinel2 as_numpy without prior Load")
    
  def as_feature(self):
    if not self.isStacked:
      self.as_numpy()
    a=self.s2.flatten(order='C')
    return _floatvector_feature(a.tolist())
#    return _dtype_feature(self.s2)
    
  def Normalize(self):
    if self.isLoaded:
      img = self.as_numpy()
      img = self.s2.astype(np.float64)
      img -= np.mean(img)
      img_std = np.std(img)
      img += img_std
      img /= img_std * 4.0
      img[np.isnan(img)] = 0.0
      img = np.clip(img, 0, 1)
      return img
    else:
        raise Exception("Sentinel2 Normalize without prior Load")

  def Crop(self, cropsize=32):
    self.image = self.Normalize()    
    # assume square baseline image which is dangerous
    self.size = self.image.shape[0]
    self.cropSize = cropsize
    self.cropsPerRow = np.floor(self.size / cropsize)
    self.nCrop = int(self.cropsPerRow * self.cropsPerRow)

  def GetCrops(self):
    return self.nCrop
        
  def GetCrop(self, crop=0):
    r = int(crop / self.cropsPerRow)
    c = int(crop % self.cropsPerRow)
    r = r * self.cropSize
    c = c * self.cropSize
    return self.image[r:r+self.cropSize,c:c+self.cropSize,:]
        
  def GetCropStats(self, crop=0):
    r = int(crop / self.cropsPerRow)
    c = int(crop % self.cropsPerRow)
    r = r * self.cropSize
    c = c * self.cropSize
    img = self.image[r:r+self.cropSize,c:c+self.cropSize,:]
    return img.mean(), img.std(), img.max(), img.min()
        
  def Plot(self, channel=None):
    if not self.isStacked:
      self.as_numpy()
    if channel is None:
      plt.imshow(self.Normalize())
      plt.show
    elif channel < 3:
      plt.imshow(self.s2[:,:,channel], cmap=gray)
      plt.show
        
  def Statistics(self, verbose=False):
    if not self.isStacked:
      self.as_numpy()
    avg = self.s2.mean()
    std = self.s2.std()
    if verbose:
      print('Shape: %d,%d,%d' % (self.s2.shape) )
      print('Mean: %f Std: %f' %(avg,std) )
    return avg, std 

  def Histogram(self, bins=10, channel=0):
    if not self.isStacked:
      self.as_numpy()
    if channel < 3:
      plt.hist(self.s2[channel], bins=bins)


# %%


class MultiEarth():
  def __init__(self, root_dir='/me2022', output_dir='/scratchX', train_split = 0.9, valid_split=0.05):
    super().__init__()

    self.rootDir = root_dir
    self.outputDir = output_dir
    self.outputPart = [None, "train", "valid", "test"]

    #self.optimizer = Adam()
    self.loss = 'mse'
    self.metrics = ['mse']
    self.monitor = 'val_loss'
    self.saveBestOnly = True
    self.saveWeightsOnly = False
    self.isTrained = False
    self.epochs = 100
    self.batchSize = 512
    self.bestEpoch = 0
    self.shardSize = 5000
    
    self.LoadSentinel1()

    self.trainPercent = train_split
    self.validPercent = valid_split
    self.testPercent = 1.0 - (train_split + valid_split)
    if self.testPercent < 0:
      raise Exception("train + valid splits must be <= 1.0")
    
    # create an array of integers in the range 0 to nExamples - 1
    x_data = np.arange(self.nExamples)
    y_data = np.copy(x_data)
    self.trainIdx, x_rem, _, y_rem = train_test_split(x_data, y_data, train_size=self.trainPercent)
    self.validIdx, self.testIdx, _, _ = train_test_split(x_rem, y_rem, train_size=self.validPercent/(self.validPercent+self.testPercent))

    self.nTrain = self.trainIdx.shape[0]
    self.nValid = self.validIdx.shape[0]
    self.nTest = self.testIdx.shape[0]

  def LoadSentinel1(self):
    f = open('%s/sentinel_vh_image_alignment_train.json'%(self.rootDir))
    self.vh = json.load(f)
    f.close()
    f = open('%s/sentinel_vv_image_alignment_train.json'%(self.rootDir))
    self.vv = json.load(f)
    f.close()
    self.example = list(self.vh)
    self.nExamples = len(self.example)

  def OutputPath(self, training_set, n):
    # the TFRecord file containing the training set
    shard = int(n / self.shardSize)
    path = '%s/%s_%d.tfr' % (self.outputDir, self.outputPart[training_set], shard)
    print(path, n)
    return path
       
  def ExampleSet(self, m):
    S1 = self.example[m]
    S2 = self.vh[S1]
    vhPath = "%s/c3/sent1/%s" % (self.rootDir, S1)
    vvPath = vhPath.replace("_VH_", "_VV_")
    s1 = Sentinel1(vhPath, vvPath)
    s1.Load()
    nS2 = int(len(S2)/3)
    example = []
    for s in range(nS2):
      r = "%s/sent2/%s" % (self.rootDir, S2[2*nS2+s])
      g = "%s/sent2/%s" % (self.rootDir, S2[1*nS2+s])
      b = "%s/sent2/%s" % (self.rootDir, S2[s])
      s2 = Sentinel2(r, g, b)
      feature = { 's1': s1.as_feature(), 's2': s2.as_feature() }
      # Create an example protocol buffer
      example.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example, nS2

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
        
  def SaveDataset(self):  
    writer = [ None,
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TRAIN, 0)),
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.VALID, 0)),
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TEST, 0)),
             ]
    counter = [0,0,0,0]
    for m in range(self.nExamples):
      if m in self.trainIdx:
        cursor = TrainingSet.TRAIN
      if m in self.validIdx:
        cursor = TrainingSet.VALID
      if m in self.testIdx:
        cursor = TrainingSet.TEST
        
      example, examples = self.ExampleSet(m)
      for e in range(examples):
        if counter[cursor] % self.shardSize == 0:
          if counter[cursor] > 0:
            writer[cursor].close()
            writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, counter[cursor]))          
        writer[cursor].write(example[e].SerializeToString())
        counter[cursor] += 1

    if counter[TrainingSet.TRAIN] > 0:
      writer[TrainingSet.TRAIN].close()
    if counter[TrainingSet.VALID] > 0:
      writer[TrainingSet.VALID].close()
    if counter[TrainingSet.TEST] > 0:
      writer[TrainingSet.TEST].close()
    
  def MakeManifest(self, manifest='selected.csv'):
    with open(manifest, 'w') as f:
      # create the csv writer
      writer = csv.writer(f)

      row = 'SAR,VIS'
      writer.writerow([row])
      for e in range(me.Examples()):
        sent1, sent2, nSent2 = me.SampleSet(e)
        for s in range(nSent2):
          avg, std = sent2[s].Statistics()
          if avg < 1000 and std < 1000:
            row='%d,%d' % (e,s)
            # write a row to the csv file
            writer.writerow([row])

  def Examples(self):
    return self.nExamples

