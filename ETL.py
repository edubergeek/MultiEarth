from enum import IntEnum
import tensorflow as tf
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow.csv as csv
import numpy as np
import os

class TrainingSet(IntEnum):
  DONOTUSE = 0
  TRAIN=1
  VALID=2
  TEST=3

# general purpose wrappers to convert Python types to flattened lists for Tensorflow
def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _dtype_feature(ndarray):
  """match appropriate tf.train.Feature class with dtype of ndarray. """
  assert isinstance(ndarray, np.ndarray)
  dtype_ = ndarray.dtype
  if dtype_ == np.float64 or dtype_ == np.float32:
    return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
  elif dtype_ == np.int64:
    return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
  else:
    raise ValueError("The input should be numpy ndarray: got {}".format(ndarray.dtype))

# Inherit and overload the TrainingExample method at a minimum
# Optionally overload the Load and init as needed
# create an array of integers in the range 0 to nExamples - 1
# self.example is a list of training examples
# self.nExamples is the number of items in the list
# SaveDataset will index self.example and write it to storage

class ETL():
  def __init__(self, root_dir, output_dir, shard_size = 5000, train_split = 0.9, valid_split=0.05):
    self.rootDir = root_dir
    self.outputDir = output_dir
    # make sure output path exists
    if not os.path.exists(self.outputDir):
      os.makedirs(self.outputDir)

    self.shardSize = shard_size
    self.outputPart = [None, "train", "valid", "live"]

    self.partitionTag = None
    self.trainPercent = train_split
    self.validPercent = valid_split
    self.testPercent = 1.0 - (train_split + valid_split)
    if self.testPercent < 0:
      raise Exception("train + valid splits must be <= 1.0")

  def SetPartitionTag(self, tag):
    self.partitionTag = tag

  def Load(self):
    f = open('%s/training_manifest.json'%(self.rootDir))
    self.manifest = json.load(f)
    f.close()
    self.example = list(self.manifest)
    self.nExamples = len(self.example)
    x_data = np.arange(self.nExamples)
    # x_data are the indices so y_data must match x_data
    y_data = np.copy(x_data)
    self.trainIdx, x_rem, _, y_rem = train_test_split(x_data, y_data, train_size=self.trainPercent)
    self.validIdx, self.testIdx, _, _ = train_test_split(x_rem, y_rem, train_size=self.validPercent/(self.validPercent+self.testPercent))

    self.nTrain = self.trainIdx.shape[0]
    self.nValid = self.validIdx.shape[0]
    self.nTest = self.testIdx.shape[0]

  def LoadExample(self, idstr):
    # return a pair of x and y examples
    x = np.fromfile(idstr+'_x')
    y = np.fromfile(idstr+'_y')
    return x, y

  def OutputPath(self, training_set, n):
    # the TFRecord file containing the training set
    shard = int(n / self.shardSize)
    if self.partitionTag is None:
      path = '%s/%s_%d.tfr' % (self.outputDir, self.outputPart[training_set], shard)
    else:
      path = '%s/%s_%s_%d.tfr' % (self.outputDir, self.outputPart[training_set], self.partitionTag, shard)
    print(path, n)
    return path

  # self.example[m] could be a file path to load
  # the end result should be an x and y training example
  def Example(self, m):
    x, y = LoadExample(self.example[m])
    return x, y

  def TrainingExample(self, m):
    x, y, id = Example(m)
    feature = { 'x': _floatvector_feature(x), 'y': _floatvector_feature(y) }
    example = []
    # Create an example protocol buffer
    example.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example, 1

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

      example, examples = self.TrainingExample(m)
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

  def Examples(self):
    return self.nExamples


class NumeraiETL(ETL):
  def __init__(self, root_dir, output_dir, shard_size = 1000, valid_split=0.01, trainfile='train.parquet', validfile='validation.parquet', testfile='live.parquet'):
    train_split = 1.0 - valid_split
    super().__init__(root_dir, output_dir, shard_size, train_split, valid_split)
    dt = datetime.now()
    self.testfile = testfile
    self.trainfile = trainfile
    self.validfile = validfile
    self.validSplit = valid_split
    self.writer = None
    self.counter = [0,0,0,0]
    self.examples = [0,0,0,0]
    self.mask = [np.ones((1)),np.ones((1)),np.ones((1)),np.ones((1))]
    #self.OpenDatasets()
    
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

  def Split(self, mode, examples):
    self.mask[mode] = np.ones((examples), dtype=np.int8)
    nValid = int(self.validSplit * examples)
    filtered = np.random.choice(examples, nValid, replace=False)
    self.mask[mode][filtered] = 0
    self.examples[TrainingSet.TRAIN] += examples - nValid
    self.examples[TrainingSet.VALID] += nValid
    self.nExamples = examples

  def Load(self, mode, reload=False):
    # load test data - either tournament or live
    if mode == TrainingSet.TRAIN:
      loadfile = self.trainfile
    elif mode == TrainingSet.VALID:
      loadfile = self.validfile
    else:
      loadfile = self.testfile
        
    table = pq.read_table(loadfile)

    # apply train valid test split
    # after this, check self.mask[mode] whether an index is in train (True) or valid (False)
    if not reload:
      self.Split(mode, table.num_rows)

    eranum = table["era"][0].as_py()
    print(eranum)
    if eranum == 'X':
      self.era = None
    else:
      self.era = table["era"].to_numpy().astype(np.int64)
    self.id = table["id"]
    
    # load features
    first_feature = 0
    last_feature  = 0
    for n in range(table.shape[1]):
      if table.column_names[n].startswith("feature_"):
        if first_feature == 0:
          first_feature = n
        last_feature = n
    X_cols = range(first_feature, last_feature+1)
    X = table.select(X_cols)
    self.X = np.empty(X.shape)
    for col in range(X.shape[1]):
      self.X[:,col] = X.column(col).to_numpy()
    
    # load targets
    first_target = 0
    last_target  = 0
    for n in range(table.shape[1]):
      if table.column_names[n].startswith("target_"):
        if first_target == 0:
          first_target = n
        last_target = n
    Y_cols = range(first_target, last_target+1)
    Y = table.select(Y_cols)
    if mode == TrainingSet.TRAIN or mode == TrainingSet.VALID:
      self.Y = np.empty(Y.shape)
      for col in range(Y.shape[1]):
        self.Y[:,col] = Y.column(col).to_numpy()
    else:
      self.Y = np.zeros(Y.shape)
   
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

  def TrainingExample(self, m):
    x, y, id, eranum = self.Example(m)
    feature = { 'x': _floatvector_feature(x), 'y': _floatvector_feature(y), 'id': _bytes_feature(id.encode('utf-8')), 'era': _int64_feature(eranum) }
    example = []
    # Create an example protocol buffer
    example.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example, eranum, 1

  def SaveDataset(self, dataset, mode, byEra=False):
    cursor = mode
    #if byEra:
    #  self.SetPartitionTag("era%d" % (self.eranum))
    # Make sure we have initialized a writer
    for m in range(self.nExamples):
      # Take an example if selected for the current mode
      if mode == TrainingSet.TEST or self.mask[dataset][m] == (mode == TrainingSet.TRAIN):
        example, eranum, examples = self.TrainingExample(m)
        # if writing TFRecord per era check for a new era
        if byEra and eranum != self.eranum:
          # if a new era then close the current era
          self.writer[cursor].close()
          # init the new era
          self.eranum = eranum
          self.SetPartitionTag("era%d" % (self.eranum))
          # todo save counter for each partition and cursor/mode
          self.counter[cursor] = 0
          self.writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, self.counter[cursor]))
  
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

