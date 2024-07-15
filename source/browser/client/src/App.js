import React, { useEffect } from 'react';
import './App.css';
import Axios from 'axios';
import io from 'socket.io-client';

const socket = io('http://127.0.0.1:8000/');

export class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      progress: 0,
      estimate_filter_markets: [],
      selected_estimate_filter_market: '',
      estimate_methods: [],
      selected_estimate_method: '',
      progress_estimate: 0,
    };

    //this.handleChange = this.handleChange.bind(this);
    this.handleSubmitFetch = this.handleSubmitFetch.bind(this);
    this.handleSubmitEstimate = this.handleSubmitEstimate.bind(this);
  }

  componentDidMount() {
    socket.on('progress', (data) => {
      this.setState({ progress: data.progress });
    });
    socket.on('progress_estimate', (data) => {
      this.setState({ progress_estimate: data.progress });
    });
    Axios.get('http://127.0.0.1:8000/estimate/filter_markets').then((response) => {
      this.setState({ estimate_filter_markets: response.data.markets, selected_estimate_filter_market: response.data.markets[0] });
    });
    Axios.get('http://127.0.0.1:8000/estimate/methods').then((response) => {
      this.setState({ estimate_methods: response.data.methods, selected_estimate_method: response.data.methods[0] });
    });
  }

  componentWillUnmount() {
    socket.off('progress');
    socket.off('progress_estimate');
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1>Investment Support with AI</h1>
          <form onSubmit={this.handleSubmitFetch}>
            <input type="submit" value="銘柄データ取得" />
          </form>
          <p>progress: {this.state.progress}%</p>
          <form onSubmit={this.handleSubmitEstimate}>
          <label>
              市場フィルタ
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
              予想手法
              <select
                value={this.state.selected_estimate_method}
                onChange={e => this.setState({ selected_estimate_method: e.target.value })}
              >
                {this.state.estimate_methods.map((method) => {
                  return <option value={method}>{method}</option>
                })}
              </select>
            </label>
            <input type="submit" value="予想データ作成" />
          </form>
          <p>progress: {this.state.progress_estimate}%</p>
        </header>
      </div>
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
}

export default App;
