import React, { useEffect } from 'react';
import Axios from 'axios';
import io from 'socket.io-client';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from 'react-bootstrap/Container';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Card from 'react-bootstrap/Card';
import Spinner from 'react-bootstrap/Spinner';

const socket = io('http://127.0.0.1:8000/');

export class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      fecth_progress_current: 0,
      fecth_progress_total: 0,
      fecth_progress_percent: 0,
      estimate_filter_markets: [],
      selected_estimate_filter_market: '',
      estimate_methods: [],
      selected_estimate_method: '',
      rebuild_estimate_model: false,
      estimate_progress_current: 0,
      estimate_progress_total: 0,
      estimate_progress_percent: 0,
      strategy_methods: [],
      selected_strategy_method: '',
      strategy_progress_current: 0,
      strategy_progress_total: 0,
      strategy_progress_percent: 0,
      evaluate_start: '2016-01-01',
      evaluate_period: 20,
      evaluate_base: 1000000,
      evaluate_gains: 100000,
      evaluate_progress_current: 0,
      evaluate_progress_total: 0,
      evaluate_progress_percent: 0,
    };

    //this.handleChange = this.handleChange.bind(this);
    this.handleSubmitFetch = this.handleSubmitFetch.bind(this);
    this.handleSubmitEstimate = this.handleSubmitEstimate.bind(this);
    this.handleSubmitStrategy = this.handleSubmitStrategy.bind(this);
    this.handleSubmitEvaluate = this.handleSubmitEvaluate.bind(this);
  }

  componentDidMount() {
    socket.on('progress_fetch', (data) => {
      var percent = 0;
      if (data.total != 0) {
        percent = Math.round((data.current / data.total) * 100);
      }
      this.setState({
        fecth_progress_current: data.current,
        fecth_progress_total: data.total,
        fecth_progress_percent: percent,
      });
    });
    socket.on('progress_estimate', (data) => {
      var percent = 0;
      if (data.total != 0) {
        percent = Math.round((data.current / data.total) * 100);
      }
      this.setState({
        estimate_progress_current: data.current,
        estimate_progress_total: data.total,
        estimate_progress_percent: percent,
      });
    });
    socket.on('progress_strategy', (data) => {
      var percent = 0;
      if (data.total != 0) {
        percent = Math.round((data.current / data.total) * 100);
      }
      this.setState({
        strategy_progress_current: data.current,
        strategy_progress_total: data.total,
        strategy_progress_percent: percent,
      });
    });
    socket.on('progress_evaluate', (data) => {
      var percent = 0;
      if (data.total != 0) {
        percent = Math.round((data.current / data.total) * 100);
      }
      this.setState({
        evaluate_progress_current: data.current,
        evaluate_progress_total: data.total,
        evaluate_progress_percent: percent,
      });
    });
    Axios.get('http://127.0.0.1:8000/estimate/filter_markets').then((response) => {
      this.setState({
        estimate_filter_markets: response.data.markets,
        selected_estimate_filter_market: response.data.markets[0]
      });
    });
    Axios.get('http://127.0.0.1:8000/estimate/methods').then((response) => {
      this.setState({
        estimate_methods: response.data.methods,
        selected_estimate_method: response.data.methods[0]
      });
    });
    Axios.get('http://127.0.0.1:8000/strategy/methods').then((response) => {
      this.setState({
        strategy_methods: response.data.methods,
        selected_strategy_method: response.data.methods[0]
      });
    });
  }

  componentWillUnmount() {
    socket.off('progress_fetch');
    socket.off('progress_estimate');
    socket.off('progress_strategy');
    socket.off('progress_evaluate');
  }

  render() {
    return (
      <Container>
          <h1>Investment Support with AI</h1>
          <Card className="mb-3">
            <Card.Title className="m-2">銘柄データ取得</Card.Title>
            <Card.Body>
              <form onSubmit={this.handleSubmitFetch}>
                <input type="submit" value="銘柄データ取得" />　
                {/* 処理中のぐるぐる（スピナー） */
                  this.state.fecth_progress_current != 0 &&
                  (this.state.fecth_progress_current != this.state.fecth_progress_total) &&
                  <Spinner animation="border" size="sm" />
                }
              </form>
              {/* プログレスバー */
                this.state.fecth_progress_current == 0 ||
                (this.state.fecth_progress_current != this.state.fecth_progress_total) ?
                <ProgressBar
                  className="my-1"
                  animated
                  now={this.state.fecth_progress_percent}
                  label={`${this.state.fecth_progress_percent}% (${this.state.fecth_progress_current}/${this.state.fecth_progress_total})`}
                /> :
                <ProgressBar
                  className="my-1"
                  variant="success"
                  now={this.state.fecth_progress_percent}
                  label={`${this.state.fecth_progress_percent}% (${this.state.fecth_progress_current}/${this.state.fecth_progress_total})`}
                />
              }
            </Card.Body>
          </Card>

          <Card className="mb-3">
            <Card.Title className="m-2">株価予想</Card.Title>
            <Card.Body>
              <form onSubmit={this.handleSubmitEstimate}>
                <p>
                  <label>
                    市場フィルタ：
                    <select
                      value={this.state.selected_estimate_filter_market}
                      onChange={e => this.setState({ selected_estimate_filter_market: e.target.value })}
                    >
                      {this.state.estimate_filter_markets.map((market) => {
                        return <option value={market}>{market}</option>
                      })}
                    </select>
                  </label>
                  <label>
                    　予想手法：
                    <select
                      value={this.state.selected_estimate_method}
                      onChange={e => this.setState({ selected_estimate_method: e.target.value })}
                    >
                      {this.state.estimate_methods.map((method) => {
                        return <option value={method}>{method}</option>
                      })}
                    </select>
                  </label>
                </p>
                <p>
                  <label>
                    <input
                      type="checkbox"
                      checked={this.state.rebuild_estimate_model}
                      onChange={e => this.setState({ rebuild_estimate_model: e.target.checked })}
                    >
                    </input>
                    （機械学習の場合）モデルを再構築する
                  </label>
                </p>
                <p>
                  <input type="submit" value="予想データ作成" />　
                  {/* 処理中のぐるぐる（スピナー） */
                    this.state.estimate_progress_current != 0 &&
                    (this.state.estimate_progress_current != this.state.estimate_progress_total) &&
                    <Spinner animation="border" size="sm" />
                  }
                </p>
              </form>
              {/* プログレスバー */
                this.state.estimate_progress_current == 0 ||
                (this.state.estimate_progress_current != this.state.estimate_progress_total) ?
                <ProgressBar
                  className="my-1"
                  animated
                  now={this.state.estimate_progress_percent}
                  label={`${this.state.estimate_progress_percent}% (${this.state.estimate_progress_current}/${this.state.estimate_progress_total})`}
                /> :
                <ProgressBar
                  className="my-1"
                  variant="success"
                  now={this.state.estimate_progress_percent}
                  label={`${this.state.estimate_progress_percent}% (${this.state.estimate_progress_current}/${this.state.estimate_progress_total})`}
                />
              }
            </Card.Body>
          </Card>

          <Card className="mb-3">
            <Card.Title className="m-2">戦略</Card.Title>
            <Card.Body>
              <form onSubmit={this.handleSubmitStrategy}>
                <p>
                  <label>
                    戦略：
                    <select
                      value={this.state.selected_strategy_method}
                      onChange={e => this.setState({ selected_strategy_method: e.target.value })}
                    >
                      {this.state.strategy_methods.map((method) => {
                        return <option value={method}>{method}</option>
                      })}
                    </select>
                  </label>
                </p>
                <p>
                  <input type="submit" value="注文データ作成" />　
                  {/* 処理中のぐるぐる（スピナー） */
                    this.state.strategy_progress_current != 0 &&
                    (this.state.strategy_progress_current != this.state.strategy_progress_total) &&
                    <Spinner animation="border" size="sm" />
                  }
                </p>
              </form>
              {/* プログレスバー */
                this.state.strategy_progress_current == 0 ||
                (this.state.strategy_progress_current != this.state.strategy_progress_total) ?
                <ProgressBar
                  className="my-1"
                  animated
                  now={this.state.strategy_progress_percent}
                  label={`${this.state.strategy_progress_percent}% (${this.state.strategy_progress_current}/${this.state.strategy_progress_total})`}
                /> :
                <ProgressBar
                  className="my-1"
                  variant="success"
                  now={this.state.strategy_progress_percent}
                  label={`${this.state.strategy_progress_percent}% (${this.state.strategy_progress_current}/${this.state.strategy_progress_total})`}
                />
              }
            </Card.Body>
          </Card>

          <Card className="mb-3">
            <Card.Title className="m-2">評価</Card.Title>
            <Card.Body>
              <form onSubmit={this.handleSubmitEvaluate}>
                <p>
                  <label>
                    開始日：
                    <input
                      value={this.state.evaluate_start}
                      onChange={e => this.setState({ evaluate_start: e.target.value })}
                      type="text"
                    >
                    </input>
                  </label>
                  <label>
                    期間（日）：
                    <input
                      value={String(this.state.evaluate_period)}
                      onChange={e => this.setState({ evaluate_period: Number(e.target.value) })}
                      type="number"
                    >
                    </input>
                  </label>
                  <label>
                    元金：
                    <input
                      value={String(this.state.evaluate_base)}
                      onChange={e => this.setState({ evaluate_base: Number(e.target.value) })}
                      type="number"
                    >
                    </input>
                  </label>
                  <label>
                    目標利益：
                    <input
                      value={String(this.state.evaluate_gains)}
                      onChange={e => this.setState({ evaluate_gains: Number(e.target.value) })}
                      type="number"
                    >
                    </input>
                  </label>
                </p>
                <p>
                  <input type="submit" value="評価" />　
                  {/* 処理中のぐるぐる（スピナー） */
                    this.state.evaluate_progress_current != 0 &&
                    (this.state.evaluate_progress_current != this.state.evaluate_progress_total) &&
                    <Spinner animation="border" size="sm" />
                  }
                </p>
              </form>
              {/* プログレスバー */
                this.state.evaluate_progress_current == 0 ||
                (this.state.evaluate_progress_current != this.state.evaluate_progress_total) ?
                <ProgressBar
                  className="my-1"
                  animated
                  now={this.state.evaluate_progress_percent}
                  label={`${this.state.evaluate_progress_percent}% (${this.state.evaluate_progress_current}/${this.state.evaluate_progress_total})`}
                /> :
                <ProgressBar
                  className="my-1"
                  variant="success"
                  now={this.state.evaluate_progress_percent}
                  label={`${this.state.evaluate_progress_percent}% (${this.state.evaluate_progress_current}/${this.state.evaluate_progress_total})`}
                />
              }
            </Card.Body>
          </Card>
      </Container>
    );
  }


  handleSubmitFetch = event => {
    var self = this;
    Axios.post('http://127.0.0.1:8000/fetch')
    .then(function(res) {
      //self.setState({ value: res.data.value });
      //alert(res.data.value);
    })
    // ブラウザによるページの更新を阻止
    event.preventDefault();
  };

  handleSubmitEstimate = event => {
    var formData = new FormData();
    formData.append('method', this.state.selected_estimate_method);
    formData.append('filter_market', this.state.selected_estimate_filter_market);
    formData.append('rebuild_model', this.state.rebuild_estimate_model);
    const customHeader = {
      headers: {
        "Content-Type": 'multipart/form-data',
      },
    };
    Axios.post('http://127.0.0.1:8000/estimate', formData, customHeader)
    .then(function(res) {
    })
    // ブラウザによるページの更新を阻止
    event.preventDefault();
  };

  handleSubmitStrategy = event => {
    var formData = new FormData();
    formData.append('estimate_method', this.state.selected_estimate_method);
    formData.append('strategy_method', this.state.selected_strategy_method);
    formData.append('base', this.state.evaluate_base);
    formData.append('filter_market', this.state.selected_estimate_filter_market);
    const customHeader = {
      headers: {
        "Content-Type": 'multipart/form-data',
      },
    };
    Axios.post('http://127.0.0.1:8000/strategy', formData, customHeader)
    .then(function(res) {
    })
    // ブラウザによるページの更新を阻止
    event.preventDefault();
  };

  handleSubmitEvaluate = event => {
    var formData = new FormData();
    formData.append('estimate_method', this.state.selected_estimate_method);
    formData.append('strategy_method', this.state.selected_strategy_method);
    formData.append('start', this.state.evaluate_start);
    formData.append('period', this.state.evaluate_period);
    formData.append('base', this.state.evaluate_base);
    formData.append('gains', this.state.evaluate_gains);
    formData.append('filter_market', this.state.selected_estimate_filter_market);
    const customHeader = {
      headers: {
        "Content-Type": 'multipart/form-data',
      },
    };
    Axios.post('http://127.0.0.1:8000/evaluate', formData, customHeader)
    .then(function(res) {
    })
    // ブラウザによるページの更新を阻止
    event.preventDefault();
  };
}

export default App;
