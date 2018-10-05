########################################
# ビューアプログラム
########################################
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# エポック数
EPOCHS = 100

# 画像の表示を行う
def display(fname):
    # 画像の読み込み
    img = Image.open(fname)
    # 表示
    img.show()

# 指定されたエポックにて、生成された画像の表示と保存を行う
# ep：エポック
# 画像ファイルの出力と保存を行う
def result(ep):
    with open('save_gimage.pkl', 'rb') as f:
        # 生成された画像を全件読み込む
        save_gimage = pkl.load(f)
        # Generatorで一度に25枚の画像を生成するため、
        # 表示エリアに5×5のパネルを準備する
        fig, axes = plt.subplots(5, 5, figsize = (28, 28))

        # zipで表示する画像と
        # 表示位置を対で取得し順に表示する
        for img, ax in zip(save_gimage[ep], axes.flatten()):
            ax.xaxis.set_visible(False)     # xスケール非表示
            ax.yaxis.set_visible(False)     # yスケール非表示
            # 画像はWidth = 28、Height = 2のため、28×28に
            # リシェイプし、グレイスケール指定で画像化する
            ax.imshow(img.reshape((28, 28)), cmap = 'gray')

        # epが-1の時は、学習最後の状態で生成された画像を
        # 対象とする
        if ep == -1:
            ep = EPOCHS - 1

        # ファイル名の編集
        fname = 'GANResult_' + format(ep, '03d') + '.png'
        print('file = ' + fname)
        # ファイル出力
        plt.savefig(fname)
        # ファイル表示
        display(fname)

# 10エポックごとの生成画像を表示
# 縦方向は10エポック単位、横方向は当該エポックで
# 生成された25枚の画像のうち、最初の5枚を表示
def history():
    with open('save_gimage.pkl', 'rb') as f:
        # エポックごとに生成された画像を全件読み込む
        save_gimage = pkl.load(f)
        # 10エポックごとに5枚の生成画像を表示するエリアを設定する
        # 画像は28×28ピクセル
        fig, axes = plt.subplots(int(EPOCHS / 10), 5, figsize = (28, 28))
        # 10エポック単位に生成画像と表示画像を順に取得しながら
        # 処理を行う
        for save_gimage, ax_row in zip(save_gimage[::10], axes):
            # 取り出したエポックには25枚の画像が含まれているため、
            # 先頭から5枚の画像を順に取り出しパネルに
            # 並べる
            for img, ax in zip(save_gimage[::1], ax_row):
                # 画像はWidth=28、Height=28のため、28×28にリシェイプし
                # グレイスケール指定で画像化する
                ax.imshow(img.reshape((28, 28)), cmap = 'gray')
                ax.xaxis.set_visible(False)  # xスケール非表示
                ax.yaxis.set_visible(False)  # yスケール非表示

        fname = 'GANHistory.png'
        print('file = ' + fname)
        plt.savefig(fname)
        display(fname)

# 学習の経過をグラフ表示で確認する
def loss():
    with open('save_loss.pkl', 'rb') as f:
        save_loss = pkl.load(f)

        # 学習損失の可視化
        fig, ax = plt.subplots()
        loss = np.array(save_loss)
        # 転置しDiscriminatorのロスは0番目の要素から、
        # Generatorのロスは1番目の要素から取得する
        plt.plot(loss.T[0], label = 'Discriminator')
        plt.plot(loss.T[1], label='Generator')
        plt.title('Loss')
        plt.legend()
        fname = 'GANLoss.png'
        print('fname = ' + fname)
        plt.savefig(fname)
        display(fname)

if __name__ == '__main__':
    args = sys.argv
    ep = 0

    if len(args) == 1:
        result(-1)
    elif args[1] == 'h':
        history()
    elif args[1] == 'l':
        loss()
    else:
        result(int(args[1]))

