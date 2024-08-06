import os
import time
import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# パリソンデータから外径/板厚情報を抽出
def extract_input_data(lines: list):
  data = []

  for line in lines:
    values = line.split()
    data.append([float(value) for value in values])

  return data

# Ls-dyna出力データから*NODEの情報を抽出
def extract_node_data(lines: list):
  data = []
  extract = False

  for line in lines:
    if line.startswith("$"):
      continue
    elif line.startswith("*NODE"):
      extract = True
      continue
    elif line.startswith("*"):
      extract = False

    if extract:
      values = line.split()
      data.append([float(value) for value in values])

  return data

# パリソンデータのみをフィルタリング
def filter_data(data: list):
  filtered_data = []

  for entry in data:
    if 500000 <= entry[0] <= 699999:
      filtered_data.append(entry)

  return filtered_data

if __name__ == "__main__":
  meshx = 100  # x方向メッシュ数
  meshy = 100  # y方向メッシュ数
  meshcut = meshx // 4  # メッシュ展開位置
  meshhalf = meshx // 2  # 2分割数

  # 予測パリソンデータの読み込み
  filepr = "/tmp/traindata/parison_prediction.dat"
  with open(filepr, "r") as f:
    lines = f.readlines()

  input_data = extract_input_data(lines)
  del input_data[0:1]
  input_data = np.repeat(input_data,meshhalf).reshape(meshy, meshx)
  input_data[:,:meshhalf] = (input_data[:,:meshhalf] - 330) / 18
  input_data[:,meshhalf:] = input_data[:,meshhalf:] - 11
  input_data = np.pad(input_data, [(3, 3), (3, 3)], "constant")
  input_data = np.reshape(input_data, (1,meshy+6,meshx+6))

  # 学習器の読み込み
  cgang = tf.keras.models.load_model('keras/cgang.keras', compile=False)

  # 予測
  prediction = cgang(input_data, training=True)
  result_data = prediction.numpy().ravel()

  # LS-dyna節点データの読み込み
  filedn = "/tmp/traindata/lsdyna_prediction.k"
  with open(filedn, "r") as f:
    lines = f.readlines()

  node_data_list = extract_node_data(lines)
  node_data = filter_data(node_data_list)
  node_data = np.array(node_data)

  # LS-DYNA形式結果ファイル出力
  with open("lsdyna_result_by_AI.k","w") as o:
    print('*KEYWORD', file=o)
    print('$TIME_VALUE =  2.0000000e+00', file=o)
    print('$STATE_NO = 42', file=o)
    print('$Output for State 42 at time = 2.00000', file=o)
    print('*ELEMENT_SHELL_THICKNESS', file=o)

    for i in range(result_data.shape[0]):
      if i < meshx*meshy:
        ielm = i + 500000
        node1 = ielm
        node2 = node1 + 1
        node3 = node2 + meshx
        node4 = node3 - 1
        if (node2-500000)%meshx == 0:
          node2 = node2 - meshx
          node3 = node3 - meshx

        if i%meshx < int(meshx-meshcut):
          j = i - int(meshx - meshcut) + meshx + meshx*3 + 21 + int(i/meshx) * 6
        else:
          j = i - int(meshx - meshcut) + meshx*3 + 21 + int(i/meshx) * 6
        re = result_data[j] * 10 + 10

        print('{:8}{:8}{:8}{:8}{:8}{:8}'.format(ielm,ielm,node1,node2,node3,node4), file=o)
        print('{:16.6e}{:16.6e}{:16.6e}{:16.6e}'.format(re,re,re,re), file=o)

    print('*NODE', file=o)

    for i in range(node_data.shape[0]):
      node = int(node_data[i][0])
      nx = node_data[i][1]
      ny = node_data[i][2]
      nz = node_data[i][3]

      print('{:8}{:16.7e}{:16.7e}{:16.7e}'.format(node,nx,ny,nz), file=o)

    print('*END', file=o)

  print(f'---- Deep Learning Prediction completed ----')
