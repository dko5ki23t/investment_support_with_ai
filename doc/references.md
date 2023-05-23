## 参考文献

### 機械学習
 * 機械学習帳

   https://chokkan.github.io/mlnote/index.html

 * LSTMで株価予測入門 [Python,Keras] | スーパーソフトウエア東京

   https://supersoftware.jp/tech/20220907/17599/

 * 【完全版】TensorFlowのインストール方法〜コマンド1発 - DeepAge

   https://deepage.net/tensorflow/2017/01/17/how-to-install-tensorflow.html

 * pyenv-win/pyenv-win

   TensorFlowをインストールする際にpyenvを使おうと思ったらWindowsはオフィシャルサポートされてなかったのでこちらを使った。

   https://github.com/pyenv-win/pyenv-win

 * PythonでKerasのLSTMを用いて、複数の情報を基に株価の予測を試してみた - リラックスした生活を過ごすために

   機械学習の説明変数を複数にする際の参考にした。

   https://relaxing-living-life.com/147/

### GPU使用
 * 機械学習時にGPUを認識してくれなくて、とっても困っている人向けの記事 | by Yutaka_kun | LSC PSD | Medium

   https://medium.com/lsc-psd/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABgpu%E3%82%92%E8%AA%8D%E8%AD%98%E3%81%97%E3%81%A6%E3%81%8F%E3%82%8C%E3%81%AA%E3%81%8F%E3%81%A6-%E3%81%A8%E3%81%A3%E3%81%A6%E3%82%82%E5%9B%B0%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%E4%BA%BA%E5%90%91%E3%81%91%E3%81%AE%E8%A8%98%E4%BA%8B-58586f1ffc0c#:~:text=%E3%81%BE%E3%81%9A%E5%88%9D%E3%82%81%E3%81%AB%E8%87%AA%E8%BA%AB%E3%81%AE%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%83%BC%E3%81%8CGPU%E3%82%92%E8%AA%8D%E8%AD%98%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%E3%82%92%E7%A2%BA%E8%AA%8D%E3%81%97%E3%81%BE%E3%81%97%E3%82%87%E3%81%86%E3%80%82%20%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%83%97%E3%83%AD%E3%83%B3%E3%83%97%E3%83%88%E3%81%A7python%E3%82%92%E8%B5%B7%E5%8B%95%E3%81%97%E4%B8%8B%E8%A8%98%E3%82%B3%E3%83%BC%E3%83%89%E3%82%92%E5%85%A5%E5%8A%9B%E3%81%97%E3%81%BE%E3%81%99%E3%80%82%20%E4%B8%8B%E3%81%AE%E3%82%88%E3%81%86%E3%81%AA%E5%87%BA%E5%8A%9B%E3%81%8C%E5%87%BA%E3%81%A6%E3%81%84%E3%82%8B%E5%A0%B4%E5%90%88%E3%80%81GPU%E3%81%AF%E8%AA%8D%E8%AD%98%E3%81%95%E3%82%8C%E3%81%A6%E3%81%84%E3%81%BE%E3%81%9B%E3%82%93%E3%80%82%20tensorflow-gpu%E3%80%81CUDA%E3%80%81cuDNN%E3%80%81pyhton%20%28Anaconda%29%E3%81%AE%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%82%92%E7%A2%BA%E8%AA%8D%E3%81%97%E3%81%A6%E3%81%84%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82,%E3%81%93%E3%82%8C%E3%81%8C%E9%81%A9%E5%88%87%E3%81%AA%E7%B5%84%E3%81%BF%E5%90%88%E3%82%8F%E3%81%9B%E3%81%A7%E3%81%AA%E3%81%84%E3%81%A8GPU%E3%81%AF%E5%8B%95%E3%81%8D%E3%81%BE%E3%81%9B%E3%82%93%E3%80%82%20%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%83%97%E3%83%AD%E3%83%B3%E3%83%97%E3%83%88%E3%81%AB%E3%81%A6python%E3%82%92%E8%B5%B7%E5%8B%95%E3%81%97%E4%B8%8B%E8%A8%98%E3%82%B3%E3%83%BC%E3%83%89%E3%82%92%E5%85%A5%E5%8A%9B%E3%80%82%20%E3%81%A7%E3%81%82%E3%82%8B%E3%81%93%E3%81%A8%E3%81%8C%E3%82%8F%E3%81%8B%E3%82%8A%E3%81%BE%E3%81%97%E3%81%9F%E3%80%82%20%E3%81%8C%E6%8E%A8%E5%A5%A8%E3%81%95%E3%82%8C%E3%81%A6%E3%81%84%E3%81%BE%E3%81%99%E3%81%AE%E3%81%A7%E3%80%81%E3%81%9D%E3%82%8C%E3%81%9E%E3%82%8C%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%82%92%E8%AA%BF%E7%AF%80%E3%81%97%E3%81%A6%E3%81%84%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82%20%E3%81%9D%E3%81%AE%E3%81%BE%E3%81%BE%E3%83%80%E3%82%A6%E3%83%B3%E3%82%B0%E3%83%AC%E3%83%BC%E3%83%89%E3%81%97%E3%81%A6%E3%82%82%E3%81%8B%E3%81%BE%E3%81%84%E3%81%BE%E3%81%9B%E3%82%93%E3%81%8C%E3%80%81%E4%BB%8A%E5%9B%9E%E3%81%AF%E4%B8%80%E5%BA%A6%E3%82%A2%E3%83%B3%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E3%81%97%E3%81%A6%E3%81%8B%E3%82%89%E3%80%81%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%82%92%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%A6%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E3%81%97%E7%9B%B4%E3%81%97%E3%81%BE%E3%81%99%E3%80%82

 * PyTorch-CUDA+cuDNN環境構築 on Windows 11 - Qiita

   ネイティブ環境での構築法で参考にした。が、結局Dockerコンテナ使うほうが良さそう、となった。

   https://qiita.com/d83yk/items/f4e5019d3f4bbeaf36dc

 * Dockerから最新のTensorflowを使う方法【Windows】 - なるぽのブログ

   https://yu-nix.com/archives/tensorflow-docker-install/

 * Tensorflowのdockerを使ってみる（docker入門） - Qiita

   https://qiita.com/mktshhr/items/d6eda04e3b4eae8fd51d

 * Windows11でWSL2＋nvidia-dockerでPyTorchを動かすのがすごすぎた

   https://blog.shikoan.com/wsl2-ndivid-docker-pytorch/

 * Windows11 + WSL2 + Docker DesktopでGPU環境を作る

   最終的にこのページの方法で環境構築するとよさそう。

   https://zenn.dev/okz/articles/83e6f899150b1e

 * WSL2＋Ubuntu 20.04でGUIアプリを動かす | AsTechLog

   WSL2(Ubuntu)をGUIで操作しようとして参考にした。成功はさせていない。

   https://astherier.com/blog/2020/08/run-gui-apps-on-wsl2/

### TensorFlow Cloud(TODO: とりあえず全部載せてるから不要なものは削除)

 * TensorFlow Cloud を使用した Keras モデルのトレーニング  |  TensorFlow Core

   https://www.tensorflow.org/guide/keras/training_keras_models_on_cloud?hl=ja

 * スタートガイド: Keras によるトレーニングと予測  |  AI Platform  |  Google Cloud

   https://cloud.google.com/ai-platform/docs/getting-started-keras?hl=ja#set_up_your_project

 * 割り当てポリシー  |  AI Platform Training  |  Google Cloud

   https://cloud.google.com/ai-platform/training/docs/quotas?hl=ja

 * 割り当てポリシー  |  AI Platform Training  |  Google Cloud

   https://cloud.google.com/ai-platform/training/docs/quotas?hl=ja#requesting_a_quota_increase

 * GCP AI platform training cannot use full GPU quota - Stack Overflow

   https://stackoverflow.com/questions/59689382/gcp-ai-platform-training-cannot-use-full-gpu-quota

### 株価データ取得

 * 株情報を取得するAPIどれが良い - Qiita

   https://qiita.com/passive-radio/items/cf3740f9601675b0a8dd


   * yahooはAPI変わったらしく、上記のdatareaderではエラーが発生するので、次項のyahooqueryを使うことにした。

     https://stackoverflow.com/questions/74831853/pandas-datareader-yahoo-daily-not-working-suddenly

 * yahooquery · PyPI

   これを使おうと思ったが、銘柄のティッカーシンボルの指定が必要で、日本株の指定（銘柄コードを指定）ができなそうだったので次項を参考にyahoo finance API（非公式）を使った。

   https://pypi.org/project/yahooquery/

 * YahooのYahoo! Finance APIを利用して株価を取得する | なんじゃもんじゃ
   
   https://nanjamonja.net/archives/1257

 * 再編成した東証市場のプライム等区分別銘柄一覧の取得 - Qiita
  
   銘柄コードの取得のために参考にした。

   https://qiita.com/higebobo/items/0ec7e243c79c0b5488bb

 * 日経平均株価をAPIで取得する2～pythonで取得できた～｜うずまき

   日経平均株価を取得するには銘柄コードを'^N225'にすればいいだけだった。

   https://note.com/uzumaki160123/n/necdbe0733219

 * Pythonから金融庁のEDINET APIを使って有価証券報告書を取得する

   決算日情報を学習に使うために参考にした。

   https://segakuin.com/python/edinet.html

 * EDINET

   https://disclosure2.edinet-fsa.go.jp/WEEK0010.aspx

### その他
 * pythonで音を鳴らす方法を詳しめに解説 - Qiita
  
   学習終了時に音で通知させるために参考にした。

   https://qiita.com/ShijiMi-Soup/items/3bbf34911f6e43ee14a3

 * otosozai.com-free音素材(wav)-

   https://otosozai.com/se1_1.html

 * Docker for Windowsで時刻ズレ対策 - Qiita

   https://qiita.com/npkk/items/ebc31451bd604bc297c1

 * [Linux(Ubuntu)]タイムゾーンを日本時間にする5つの方法まとめ

  時刻ずれてると思ったらタイムゾーンが日本になってなかった(アホ)。

  https://zenn.dev/kumamoto/articles/51bf0893620f0c
  