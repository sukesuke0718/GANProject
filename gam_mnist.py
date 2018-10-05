######################################
# 画像生成プログラム
######################################
import tensorflow as tf
# MNISTデータセット取得のための準備
from tensorflow.examples.tutorials.mnist import input_data
import datetime as dt
import numpy as np
# 学習過程で生成される画像の保管にはplk形式を使う
import pickle as pkl
# 活性化関数を指定するときに使用
from functools import partial

# 各種パラメータ

# エポック数。ビューアプログラムで定義
# EPOCHSと同じにする
EPOCHS = 100
# バッチサイズ
BATCH_SIZE = 100
# 学習率
LEARNING_RATE = 0.001
# 活性化関数のハイパーパラメータ設定
ALPHA = 0.01


# 生成モデルを作る関数の定義
def generator(randomData, alpha, reuse = False):
    with tf.variable_scope('GAN/generator', reuse = reuse):
        # 隠れ層
        h1 = tf.layers.dense(randomData, 256, activation = partial(tf.nn.leaky_relu, alpha = alpha))
        o1 = tf.layers.dense(h1, 784, activation = None)
        # 活性化関数 tanh
        img = tf.tanh(o1)

    return img

# 識別モデルを作る関数の定義
def discriminator(img, alpha, reuse = False):
    with tf.variable_scope('GAN/discriminator', reuse = reuse):
        # 隠れ層
        h1 = tf.layers.dense(img, 128, activation = partial(tf.nn.leaky_relu, alpha = alpha))
        # 出力層
        D_logits = tf.layers.dense(h1, 1, activation = None)
        # 活性化関数
        D = tf.nn.sigmoid(D_logits)

    return D, D_logits

if __name__ == '__main__':
    # 処理開始時刻の取得
    tstamp_s = dt.datetime.now().strftime("%H:%M:%S")
    # MNISTデータでっとのダウンロード
    mnist = input_data.read_data_sets('MNIST_DataSet')

    # プレースホルダー
    # 本物画像データをバッチサイズ分
    # 保存するプレースホルダーを準備する
    ph_realData = tf.placeholder(tf.float32, (BATCH_SIZE, 784))
    # 100次元の一様乱数を補完するプレースホルダーを準備
    # 確保するサイズは、学習時はバッチサイズの100件、
    # 各エポックでの画像生成時は25件と、動的に変わるため、
    # Noneを指定し、実行時にサイズを決定するようにする
    ph_randamData = tf.placeholder(tf.float32, (None,100))

    # 一様乱数を与えて画像を生成
    gimage = generator(ph_randamData, ALPHA)
    # 本物の画像を与えて画像を生成
    real_D, real_D_logits = discriminator(ph_realData,ALPHA)
    # 生成画像を与えて判定結果を取得
    fake_D, fake_D_logits = discriminator(gimage, ALPHA, reuse = True)

    # 損失関数の実装
    # 本物画像との誤差を
    # クロスエントロピーの平均として取得
    d_real_xentropy =  \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels = tf.ones_like(real_D))
    loss_real = tf.reduce_mean(d_real_xentropy)
    # 生成画像との誤差を
    # クロスエントロピーの平均として取得
    d_fake_xentropy = \
    tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_D_logits, labels = tf.zeros_like(fake_D))
    loss_fake = tf.reduce_mean(d_fake_xentropy)
    # Discriminstorの誤差を本物画像、生成画像における誤差を
    # 合計した値とる
    d_loss = loss_real + loss_fake
    # Generatorの誤差を取得
    g_xentropy = \
    tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_D_logits, labels = tf.ones_like(fake_D))
    g_loss = tf.reduce_mean(g_xentropy)

    # 学習によって最適化を行うパラメータを
    # tf.trainable_variablesから一括して取得する。その際に、
    # Discriminator用(d_training_parameter)、Generator用(g_training_parameter)
    # と分けて、それぞれのネットワークを最適化していく必要があるため、
    # ネットワーク定義時に指定したスコープの名前を取得して
    # とりわけを行う
    # discrimatorの最適化を行う学習パラメータを取得
    # （一旦、trainVarにとりわけてから格納）
    d_training_parameter = \
    [trainVar for trainVar in tf.trainable_variables() if 'GAN/discriminator/' in trainVar.name]
    # generatorの最適化を行う学習パラメータを取得
    g_training_parameter = \
    [trainVar for trainVar in tf.trainable_variables() if 'GAN/generator/' in trainVar.name]

    # オプティマイザで学習パラメータの
    # 最適化を行う
    # 一括取得したDiscrimninatorのパラメータを更新
    d_optimize = \
    tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(d_loss, var_list = d_training_parameter)
    # 一括取得したGeneratorのパラメータを更新
    g_optimize = \
    tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(g_loss, var_list = g_training_parameter)

    batch = mnist.train.next_batch(BATCH_SIZE)

    # 途中経過の保存する変数を定義
    save_gimage = []
    save_loss = []

    # 学習処理の実装
    with tf.Session() as sess:
        # 変数の初期化
        sess.run(tf.global_variables_initializer())

        # EPOCHS数分繰り返す
        for e in range(EPOCHS):
            # バッチサイズ100
            for i in range(mnist.train.num_examples//BATCH_SIZE):
                batch = mnist.train.next_batch(BATCH_SIZE)
                batch_images = batch[0].reshape((BATCH_SIZE, 784))
                # genaratorにて活性化関数tanhを使用したため、
                # レンジを合わせる
                batch_images = batch_images * 2 - 1
                # generatorに渡す①一様分布のランダムノイズを生成
                # 値は-1～1まで、サイズはbatch_size * 100
                batch_z = np.random.uniform(-1, 1, size = (BATCH_SIZE, 100))
                # 最適化計算・パラメータ更新を行う
                # Discriminatorの最適化に使うデータ群をfeed_dictで与える
                sess.run(d_optimize, feed_dict = {ph_realData:batch_images, ph_randamData: batch_z})
                # Generatorの最適化と最適化に使うデータ群を
                # feed_dictで与える
                sess.run(g_optimize, feed_dict = {ph_randamData: batch_z})

            # トレーニングのロスを記録
            train_loss_d = sess.run(d_loss, {ph_randamData: batch_z, ph_realData: batch_images})
            # evalはgeneratorのロス（g_loss）を出力すつ命令
            train_loss_g = g_loss.eval({ph_randamData:batch_z})

            # 学習過程の表示
            print('{0} Epoch = {1}/{2}, DLoss={3:.4F}, ' \
                  'GLoss={4:.4F}'.format(dt.datetime.now().strftime("%H:%M:%S"), e + 1, \
                EPOCHS, train_loss_d, train_loss_g))

            # lossを格納するためのリストに追加する
            # train_loss_d, train_loss_gをセットでリストに追加し
            # 後で可視化できるようにする
            save_loss.append((train_loss_d, train_loss_g))

            # 学習途中の生成モデルで画像を生成して保存する
            # 一様乱数データを25個生成して、そのデータを使って画像を生成し、保存する
            randomData = np.random.uniform(-1, 1, size = (25,100))
            # gen_samplesに現時点のモデルで作ったデータを読ませておく
            # ノイズ、サイズ、ユニット数（128）、reuseは状態維持、
            # データはrandomDataとしてfeed_dictに指定
            gen_samples = \
            sess.run(generator(ph_randamData, ALPHA, True), feed_dict = {ph_randamData: randomData})
            # 生成画像を保存
            save_gimage.append(gen_samples)

        # pkl形式で生成画像を保存
        with open('save_gimage.pkl', 'wb') as f:
            pkl.dump(save_gimage, f)

        # 各エポックで得た損失関数の値を保存
        with open('save_loss.pkl', 'wb') as f:
            pkl.dump(save_loss, f)

        # 処理終了時刻の取得
        tstamp_e = dt.datetime.now().strftime("%H:%M:%S")

        time1 = dt.datetime.strptime(tstamp_s, "%H:%M:%S")
        time2 = dt.datetime.strptime(tstamp_e, "%H:%M:%S")

        # 処理時間を表示
        print("開始：{0}、終了：{1}、処理時間：{2}".format(tstamp_s, tstamp_e, (time2 - time1)))
