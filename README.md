# investment_support_with_ai

## 環境構築のためにやったこと(Docker環境)

* ※のちにWindows11のネイティブ環境で動作を確認できたため次項を参照。

* 基本は以下ページ通り。必要なものはpipでインストール

  https://zenn.dev/okz/articles/83e6f899150b1e

~~ * 最新のTensorFlowだと警告がたくさん出るので、ダウングレード

  ```
  pip install install tensorflow==2.11
  ```

  参考：https://discuss.tensorflow.org/t/you-must-feed-a-value-for-placeholder-tensor-gradients-split-dim-with-dtype-int32/15712/7 ~~ 

* cuDNNのダウンロード・インストール

  参考：https://super-vitality.com/tensorflow-nvidia-gpu/

  参考：https://www.kkaneko.jp/tools/ubuntu/ubuntu_cudnn.html
  
  ```
  apt -y update
  apt -y install libcudnn8 libcudnn8-dev
  ```

* ntpdateのインストール

  ```
  apt update
  apt install ntpdate
  ```

* タイムゾーンの変更

  参考：https://zenn.dev/kumamoto/articles/51bf0893620f0c

  ```
  cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
  ```

### 注意事項

* 各スクリプト実行前に以下を実行して時刻を合わせる(TODO: timedatectlやchronyのサービスで自動で更新させる)

```
ntpdate ntp.nict.jp
```

もしくは他のNTPのサーバーでも可。

```
ntpdate ntp.jst.mfeed.ad.jp
```

### Dockerの起動コマンド

```
docker run --gpus all -it --privileged -v C:\Users\dko5k\Documents\git\investment_support_with_ai:/home/git/investment_support_with_ai investment_gpu:base
```

* `--privileged` をつけることで`ntpdate`を実行できる権限を付与する。(https://qiita.com/npkk/items/ebc31451bd604bc297c1)

## 環境構築のためにやったこと(Windows11ネイティブ環境)

* 基本は以下ページの通りに実施。各ツールのバージョンがややこしいので後述を参照。

  https://www.tensorflow.org/install/gpu?hl=ja#software_requirements

* TensorRTは未インストール。

### 動作確認環境：各ツール等のバージョン

#### ハードウェア

* プロセッサ

   * Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz

* メモリ

   * 16.0GB

* GPU(2つ搭載、TensorFlowではNVIDIAのものを使ってる・・・はず)

   * Intel(R) UHD Graphics

   * NVIDIA GeForce RTX 3050 Laptop GPU

#### ソフトウェア

https://www.tensorflow.org/install/source?hl=ja#gpu を参考に、TensorFlowに合わせて各種ドライバのバージョンを選択した。今回はTensorFlow 2.6.0で環境構築した。（もっと新しくてもいいかも）

* Windows 11 Home 22H2

* NVIDIA GPU ドライバ(https://www.nvidia.com/download/index.aspx?lang=en-us)

   * 531.68(上記ページで適切に選択すればたぶん多少バージョン違っても大丈夫)

   * バージョンを確認するには、デスクトップで右クリック->NVIDIAコントロールパネル

* CUDAツールキット(https://developer.nvidia.com/cuda-11.2.0-download-archive)

   * 11.2.0

   * インストーラはサイズが約3GBある

* cuDNN SDK(https://developer.nvidia.com/rdp/cudnn-download)

   * v8.9.1 (May 5th, 2023), for CUDA 11.x (Local Installer for Windows (Zip))

   * zip解凍後、C:\tools\cudaに展開（PATH通せばどこでもいいが。https://www.tensorflow.org/install/gpu?hl=ja#windows_setup）

* python 3.9.13

* pythonパッケージ

  以下に示すパッケージをpipでインストールすればOKのはず（python実行時に足りないとエラー出たら都度pipする）

  パッケージ同士の依存関係のせいでダウングレードが必要なものあり。バージョンを指定したものは太字にて示す。

   * tensorflow - **2.6.0**

   * scikit-learn - 1.2.2

   * keras - **2.6.0** (https://stackoverflow.com/questions/72255562/cannot-import-name-dtensor-from-tensorflow-compat-v2-experimental)

   * protobuf - **3.20.1** (https://masaki-note.com/2022/05/29/protobuf_downgrade/)

   * pandas - **1.3.5**

   * numpy - **1.19.5** (tensorflow, pandas, numpyの依存関係がけっこうクセ者)

   * tqdm - 4.65.0

   * yahoo-finance-api2 - 0.0.12
