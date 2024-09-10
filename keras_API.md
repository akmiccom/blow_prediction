#  Keras: TensorFlow の高レベル API

## Keras API コンポーネント
### Layer
- レイヤーの作成

```Python
tf.keras.layers.Layer
```

### Model
- トレーニングと評価メソッドが組み込まれている
  - 一定したエポック数でモデルをトレーニング
  - 入力サンプルに対して出力予測を生成
  - モデルの損失と始業の値を返す
  - メソッドで構成される

```Python
tf.keras.Model
    tf.keras.Model.fit
    tf.keras.Model.predict
    tf.keras.Model.eveluate
    tf.keras.Model.compile
```

