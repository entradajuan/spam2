import tensorflow as tf
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

!pip install -q transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/gdrive')
from pickle import load
np.random.seed(0)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(torch.cuda.get_device_name(0))

lines = io.open('data/SMSSpamCollection').read().strip().split('\n')
data = []
count = 0
for e in lines:
  label, text = e.split('\t')
  if (label.lower().strip() =='spam'):
    data.append((1, text.strip()))
  else:
    data.append((0, text.strip()))

df = pd.DataFrame(data, columns=['spam', 'text'])
print(df.head())
print(df.shape)
print(df.isna().sum())
print(df[].value_counts())

