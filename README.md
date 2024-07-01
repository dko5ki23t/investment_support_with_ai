# investment_support_with_ai

## 概要

準備中

## 事前準備

- [J-Quants](https://jpx-jquants.com/)の登録（無料プランでも OK）

## 使い方

1.

## 注意事項

## 環境構築のためにやったこと(Docker 環境)

- ※のちに Windows11 のネイティブ環境で動作を確認できたため次項を参照。

- 基本は以下ページ通り。必要なものは pip でインストール

  https://zenn.dev/okz/articles/83e6f899150b1e

~~ \* 最新の TensorFlow だと警告がたくさん出るので、ダウングレード

```
pip install install tensorflow==2.11
```

参考：https://discuss.tensorflow.org/t/you-must-feed-a-value-for-placeholder-tensor-gradients-split-dim-with-dtype-int32/15712/7 ~~

- cuDNN のダウンロード・インストール

  参考：https://super-vitality.com/tensorflow-nvidia-gpu/

  参考：https://www.kkaneko.jp/tools/ubuntu/ubuntu_cudnn.html

  ```
  apt -y update
  apt -y install libcudnn8 libcudnn8-dev
  ```

- ntpdate のインストール

  ```
  apt update
  apt install ntpdate
  ```

- タイムゾーンの変更

  参考：https://zenn.dev/kumamoto/articles/51bf0893620f0c

  ```
  cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
  ```

### 注意事項

- 売買戦略の評価は、あくまで当該売買指示をその日の一番初めから実行した場合で考えている。本プログラムで作成した売買指示も、該当の日時で市場開始時から指示通りの注文をしておくことで評価結果と似た結果が得られる。（TODO:文章もっと正しく記述）

- 各スクリプト実行前に以下を実行して時刻を合わせる(TODO: timedatectl や chrony のサービスで自動で更新させる)

```
ntpdate ntp.nict.jp
```

もしくは他の NTP のサーバーでも可。

```
ntpdate ntp.jst.mfeed.ad.jp
```

### Docker の起動コマンド

```
docker run --gpus all -it --privileged -v C:\Users\dko5k\Documents\git\investment_support_with_ai:/home/git/investment_support_with_ai investment_gpu:base
```

- `--privileged` をつけることで`ntpdate`を実行できる権限を付与する。(https://qiita.com/npkk/items/ebc31451bd604bc297c1)

## 環境構築のためにやったこと(Windows11 ネイティブ環境)

- 基本は以下ページの通りに実施。各ツールのバージョンがややこしいので後述を参照。

  https://www.tensorflow.org/install/gpu?hl=ja#software_requirements

- TensorRT は未インストール。

- Python 環境構築は以下の手順で。

  - git を入れる(pyenv 入れるため)

    https://git-scm.com/download/win

  - pyenv をインストール

    https://zenn.dev/lot36z/articles/1c734bde03677c

  - pyenv で所望の pytthon バージョンをインストール

    https://qiita.com/twipg/items/75fc9428e4c33ed429c0

### 動作確認環境：各ツール等のバージョン

#### ハードウェア

- プロセッサ

  - Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz

- メモリ

  - 16.0GB

- GPU(2 つ搭載、TensorFlow では NVIDIA のものを使ってる・・・はず)

  - Intel(R) UHD Graphics

  - NVIDIA GeForce RTX 3050 Laptop GPU

#### ソフトウェア

https://www.tensorflow.org/install/source?hl=ja#gpu を参考に、TensorFlow に合わせて各種ドライバのバージョンを選択した。今回は TensorFlow 2.6.0 で環境構築した。（もっと新しくてもいいかも）

- Windows 11 Home 22H2

- NVIDIA GPU ドライバ(https://www.nvidia.com/download/index.aspx?lang=en-us)

  - 531.68(上記ページで適切に選択すればたぶん多少バージョン違っても大丈夫)

  - バージョンを確認するには、デスクトップで右クリック->NVIDIA コントロールパネル

- CUDA ツールキット(https://developer.nvidia.com/cuda-11.2.0-download-archive)

  - 11.2.0

  - インストーラはサイズが約 3GB ある

- cuDNN SDK(https://developer.nvidia.com/rdp/cudnn-download)

  - v8.9.1 (May 5th, 2023), for CUDA 11.x (Local Installer for Windows (Zip))

  - zip 解凍後、C:\tools\cuda に展開（PATH 通せばどこでもいいが。https://www.tensorflow.org/install/gpu?hl=ja#windows_setup）

- python 3.9.13

- python パッケージ

  以下に示すパッケージを pip でインストールすれば OK のはず（python 実行時に足りないとエラー出たら都度 pip する）

  パッケージ同士の依存関係のせいでダウングレードが必要なものあり。バージョンを指定したものは太字にて示す。

  - tensorflow - **2.6.0**

  - scikit-learn - 1.2.2

  - keras - **2.6.0** (https://stackoverflow.com/questions/72255562/cannot-import-name-dtensor-from-tensorflow-compat-v2-experimental)

  - protobuf - **3.20.3** (https://masaki-note.com/2022/05/29/protobuf_downgrade/)

  - pandas - **1.3.5**

  - numpy - **1.19.5** (tensorflow, pandas, numpy の依存関係がけっこうクセ者)

  - tqdm - 4.65.0

  - yahoo-finance-api2 - 0.0.12

### その他

- 環境変数 Path に以下を追加(https://github.com/tensorflow/tensorflow/issues/50273)

  ```
  C:\Program Files\NVIDIA Corporation\Nsight Systems 2020.4.3\target-windows-x64
  ```
