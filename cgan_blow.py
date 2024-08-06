import os
import time
import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs("keras/", exist_ok=True)
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pathlib
from contextlib import redirect_stdout
from tensorflow.python.framework import tensor_util
from tensorflow.core.util import event_pb2

# パリソンデータから外径/板厚情報を抽出
def extract_input_data(lines: list):
  data = []

  for line in lines:
    values = line.split()
    data.append([float(value) for value in values])

  return data

# Ls-dyna出力データから*ELEMENT_SHELL_THICKNESSの情報を抽出
def extract_result_data(lines: list):
  data = []
  extract = False

  for line in lines:
    if line.startswith("$"):
      continue
    elif line.startswith("*ELEMENT_SHELL_THICKNESS"):
      extract = True
      continue
    elif line.startswith("*"):
      extract = False

    if extract:
      values = line.split()
      data.append([float(value) for value in values])

  id_list_list = [sublist for index, sublist in enumerate(data) if index % 2 == 0]
  id_list = [sublist[0] for sublist in id_list_list]

  result_list_list = [sublist for index, sublist in enumerate(data) if index % 2 != 0]
  result_list = [sum(sublist)/len(sublist) for sublist in result_list_list]

  result_data_list = [[x, y] for x, y in zip(id_list, result_list)]

  return result_data_list

# パリソンデータのみをフィルタリング
def filter_data(data: list):
  filtered_data = []

  for entry in data:
    if 500000 <= entry[0] <= 699999:
      filtered_data.append(entry)

  return filtered_data

# データの読み込み
def load(data):
  m = tf.shape(data)[1]
  m = m // 2
  input_data = data[:,:m,:]
  result_data = data[:,m:,:]

  return input_data, result_data

# 畳み込み
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# 逆畳み込み
def upsample(filters, size, apply_dropout=False):
  init = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                             kernel_initializer=init, use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.05))

  result.add(tf.keras.layers.ReLU())

  return result

# 生成器関数
def funcg():
  inp = tf.keras.layers.Input(shape=[meshy+6, meshx+6, 1])

  std = [
    downsample(64, 4, apply_batchnorm=False),
    downsample(128, 4),
    downsample(256, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
  ]

  stu = [
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4),
    upsample(256, 4),
    upsample(128, 4),
    upsample(64, 4),
  ]

  init = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same',
                                         kernel_initializer=init, activation='tanh')

  rlt = tf.keras.layers.Resizing(256, 256, interpolation='bilinear')(inp)

  stk = []
  for dw in std:
    rlt = dw(rlt)
    stk.append(rlt)

  stk = reversed(stk[:-1])

  for up, sk in zip(stu, stk):
    rlt = up(rlt)
    rlt = tf.keras.layers.Concatenate()([rlt, sk])

  rlt = last(rlt)
  rlt = tf.keras.layers.Resizing(meshy+6, meshx+6, interpolation='bilinear')(rlt)

  return tf.keras.Model(inputs=inp, outputs=rlt)

# 生成器の損失関数
def lsg(dset, gdata, rlt):
  lsg1 = ls(tf.ones_like(dset), dset)
  lsg2 = tf.reduce_mean(tf.abs(rlt - gdata))
  lsgs = lsg1 + lsg2 * 100

  return lsgs

# 識別器関数
def funcd():
  init = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[meshy+6, meshx+6, 1])
  out = tf.keras.layers.Input(shape=[meshy+6, meshx+6, 1])
  set = tf.keras.layers.concatenate([inp, out], axis=2)
  rlt = tf.keras.layers.Resizing(256, 512, interpolation='bilinear')(set)
  rlt = downsample(64, 4, False)(rlt)
  rlt = downsample(128, 4)(rlt)
  rlt = downsample(256, 4)(rlt)
  rlt = tf.keras.layers.Resizing(32, 32, interpolation='bilinear')(rlt)
  rlt = tf.keras.layers.ZeroPadding2D()(rlt)
  rlt = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=init,use_bias=False)(rlt)
  rlt = tf.keras.layers.BatchNormalization()(rlt)
  rlt = tf.keras.layers.LeakyReLU()(rlt)
  rlt = tf.keras.layers.ZeroPadding2D()(rlt)
  rlt = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=init)(rlt)

  return tf.keras.Model(inputs=set, outputs=rlt)

# 識別器の損失関数
def lsd(ddata, dset):
  lsd1 = ls(tf.ones_like(ddata), ddata)
  lsd2 = ls(tf.zeros_like(dset), dset)
  lsds = lsd1 + lsd2

  return lsds

# 学習
@tf.function
def tr(inp, rlt, kep):
  with tf.GradientTape() as tpg, tf.GradientTape() as tpd:
    gdata = cgang(inp, training=True)
    setdata = tf.keras.layers.concatenate([inp, rlt], axis=2)
    ddata = cgand(setdata, training=True)
    gset = tf.keras.layers.concatenate([inp, gdata], axis=2)
    dset = cgand(gset, training=True)
    lsgs = lsg(dset, gdata, rlt)
    lsds = lsd(ddata, dset)

  grg = tpg.gradient(lsgs,cgang.trainable_variables)
  grd = tpd.gradient(lsds,cgand.trainable_variables)
  opg.apply_gradients(zip(grg,cgang.trainable_variables))
  opd.apply_gradients(zip(grd,cgand.trainable_variables))

  with record.as_default():
    tf.summary.scalar('loss_g', lsgs, step=kep)
    tf.summary.scalar('loss_d', lsds, step=kep)

# 損失関数描画
def lossplot(data,lpath):
  data = np.reshape(data, (-1, 2)).transpose()
  
  plt.figure(figsize=(12,8))
  plt.plot(data[0],data[1])
  plt.xlim(-5,100)
  plt.ylim(0,25)
  plt.xlabel('Step',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.tick_params(labelsize=20)
  plt.savefig(lpath)

# メインプログラム
if __name__ == "__main__":
  ntr = 10  # 教師データ数
  epoch = 100  # エポック数
  step = ntr * epoch  # ステップ数
  meshx = 100  # x方向メッシュ数
  meshy = 100  # y方向メッシュ数
  meshcut = meshx // 4  # メッシュ展開位置
  meshhalf = meshx // 2  # 2分割数
  setdata = np.empty([0, meshy+6, meshx*2+12, 1])

  for k in range(ntr):

    # パリソンデータの読み込み
    filepr = "/tmp/traindata/parison" + str(k+1) + ".dat"
    with open(filepr, "r") as f:
      lines = f.readlines()

    input_data = extract_input_data(lines)
    del input_data[0:1]
    input_data = np.repeat(input_data,meshhalf).reshape(meshy, meshx)
    input_data[:,:meshhalf] = (input_data[:,:meshhalf] - 330) / 18
    input_data[:,meshhalf:] = input_data[:,meshhalf:] - 11
    input_data = np.pad(input_data, [(3, 3), (3, 3)], "constant")

    # LS-dyna出力ファイルの読み込み
    filedn = "/tmp/traindata/lsdyna" + str(k+1) + ".k"
    with open(filedn, "r") as f:
      lines = f.readlines()

    result_data_list = extract_result_data(lines)
    filtered_result_list = filter_data(result_data_list)
    result_data =  [sublist[1] for sublist in filtered_result_list]
    result_data = np.reshape(result_data, (meshy, meshx))
    result_data = np.block([result_data[:,meshx-meshcut:],result_data[:,:meshx-meshcut]])
    result_data = (result_data - 10) / 10
    result_data = np.pad(result_data, [(3, 3), (3, 3)], "constant")
    data = np.block([input_data,result_data]).reshape(1,meshy+6,meshx*2+12,1)
    setdata = np.concatenate([setdata, data], 0).astype('float32')

  # 教師データの前処理
  trdata = tf.data.Dataset.from_tensor_slices(setdata)
  trdata = trdata.map(load)
  trdata = trdata.shuffle(ntr)
  trdata = trdata.batch(1)

  # 生成器、識別器の呼び出し
  cgang = funcg()
  cgand = funcd()
  opg = tf.keras.optimizers.RMSprop(5e-4, rho=0.95)
  opd = tf.keras.optimizers.RMSprop(1e-5, rho=0.8)
  ls = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  # ログファイル書き出し
  now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  stdtime = now
  sttime = time.time()
  print(f'')
  print(f'---------------------------------------------------------------------------------------------')
  print(f'    Deep Learning by Conditional Generative Adversarial Networks      {now}')
  print(f'')
  print(f'       Training dataset {ntr}')
  print(f'       Epoch {epoch}')
  print(f'       Total step {step}')
  print(f'       Mesh size {meshx} X {meshy}')
  print(f'---------------------------------------------------------------------------------------------')
  print(f'')

  o = open("mes.dat","w")
  print(f'', file=o)
  print(f'---------------------------------------------------------------------------------------------', file=o)
  print(f'    Deep Learning by Conditional Generative Adversarial Networks      {now}', file=o)
  print(f'', file=o)
  print(f'       Training dataset {ntr}', file=o)
  print(f'       Epoch {epoch}', file=o)
  print(f'       Total step {step}', file=o)
  print(f'       Mesh size {meshx} X {meshy}', file=o)
  print(f'---------------------------------------------------------------------------------------------', file=o)
  print(f'', file=o)

  log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  record = tf.summary.create_file_writer("loss_history/event_file/" + log_time)

  # 学習
  init = time.time()
  pvt = init

  for kep, (inp, rlt) in trdata.repeat().take(step+1).enumerate():
    tr(inp, rlt, kep)

    if kep != 0 and (kep) % 10 == 0:
      crt = time.time()
      now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      print(f'  Step {kep}   dt {crt-pvt:.2f} sec   total time {crt-init:.2f} sec       {now}')
      print(f'  Step {kep}   dt {crt-pvt:.2f} sec   total time {crt-init:.2f} sec       {now}', file=o)
      pvt = time.time()

    if kep == step:
      print(f'')
      print(f'', file=o)
      print(f'       Step end reached')
      print(f'       Step end reached', file=o)
      print(f'')
      print(f'')
      print(f'', file=o)
      print(f'', file=o)

  # 損失関数プロット
  ev = "loss_history/event_file/" + log_time

  og = open("./loss_history/cgan_loss_g.dat","w")
  od = open("./loss_history/cgan_loss_d.dat","w")
  print(f'step  loss_g', file=og)
  print(f'step  loss_d', file=od)
  lgpath = "./loss_history/cgan_loss_g.png"
  ldpath = "./loss_history/cgan_loss_d.png"

  datag = []
  datad = []

  for fname in os.listdir(ev):
    path = os.path.join(ev, fname)
    ldataset = tf.data.TFRecordDataset(path)

    for ldata in ldataset:
      event = event_pb2.Event.FromString(ldata.numpy())
      for value in event.summary.value:
        t = tf.make_ndarray(value.tensor)

        if value.tag == 'loss_g':
          print(event.step, t, file=og)
          datag = np.append(datag, event.step)
          datag = np.append(datag, t)

        if value.tag == 'loss_d':
          print(event.step, t, file=od)
          datad = np.append(datad, event.step)
          datad = np.append(datad, t)

  lossplot(datag,lgpath)
  lossplot(datad,ldpath)

  og.close()
  od.close()

  # 出力
  print(f'---------------------------------------------------------------------------------------------')
  print(f'    Convolutional Neural Network Summary')
  print(f'---------------------------------------------------------------------------------------------')
  print(f'')

  print(f'---------------------------------------------------------------------------------------------', file=o)
  print(f'    Convolutional Neural Network Summary', file=o)
  print(f'---------------------------------------------------------------------------------------------', file=o)
  print(f'', file=o)

  cgang.summary()
  with redirect_stdout(o):
    cgang.summary()

  print(f'')
  print(f'', file=o)

  cgand.summary()
  with redirect_stdout(o):
    cgand.summary()

  cgang.save('./keras/cgang.keras')
  cgand.save('./keras/cgand.keras')

  now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  edtime = time.time()
  eptime = edtime - sttime
  hour = int(eptime / 3600)
  minute = int((eptime / 60) % 60)
  second = int(eptime % 60)

  print(f'')
  print(f'')
  print(f'  Start time    {stdtime}')
  print(f'  End time      {now}')
  print(f'  Elapsed time     {eptime:.2f} seconds  ( {hour} hour  {minute} minutes  {second} seconds )')
  print(f'')
  print(f'---- Normal Termination ----                       {now}')
  print(f'')

  print(f'', file=o)
  print(f'', file=o)
  print(f'  Start time    {stdtime}', file=o)
  print(f'  End time      {now}', file=o)
  print(f'  Elapsed time     {eptime:.2f} seconds  ( {hour} hour  {minute} minutes  {second} seconds )', file=o)
  print(f'', file=o)
  print(f'---- Normal Termination ----                       {now}', file=o)
  print(f'', file=o)

  o.close()
