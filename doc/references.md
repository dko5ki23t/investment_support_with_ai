## 参考文献

### 機械学習
 * 機械学習帳

   https://chokkan.github.io/mlnote/index.html

 * LSTMで株価予測入門 [Python,Keras] | スーパーソフトウエア東京

   https://supersoftware.jp/tech/20220907/17599/

 * 【完全版】TensorFlowのインストール方法〜コマンド1発 - DeepAge

   https://deepage.net/tensorflow/2017/01/17/how-to-install-tensorflow.html

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
  