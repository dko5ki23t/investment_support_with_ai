# investment_support_with_ai

## 環境構築のためにやったこと

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

## 注意事項

* 各スクリプト実行前に以下を実行して時刻を合わせる(TODO: timedatectlやchronyのサービスで自動で更新させる)

```
ntpdate ntp.nict.jp
```

もしくは他のNTPのサーバーでも可。

```
ntpdate ntp.jst.mfeed.ad.jp
```

## Dockerの起動コマンド

```
docker run --gpus all -it --privileged -v C:\Users\dko5k\Documents\git\investment_support_with_ai:/home/git/investment_support_with_ai investment_gpu:base
```

* `--privileged` をつけることで`ntpdate`を実行できる権限を付与する。(https://qiita.com/npkk/items/ebc31451bd604bc297c1)
