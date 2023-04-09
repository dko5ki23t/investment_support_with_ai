## 参考文献

### 機械学習
 * 機械学習帳

   https://chokkan.github.io/mlnote/index.html

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
  