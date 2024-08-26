import React, { useContext } from 'react';
import Axios from 'axios';
import io from 'socket.io-client';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from 'react-bootstrap/Container';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { useAccordionButton } from 'react-bootstrap/AccordionButton';
import Card from 'react-bootstrap/Card';
import Accordion from 'react-bootstrap/Accordion';
import AccordionContext from 'react-bootstrap/AccordionContext';
import Spinner from 'react-bootstrap/Spinner';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

const socket = io('http://127.0.0.1:8000/');

function CustomToggleCol({ children, eventKey, md }) {
  const decoratedOnClick = useAccordionButton(eventKey, function() {},
  );

  return (
    <Col onClick={decoratedOnClick} md={md}>
      {children}
    </Col>
  );
}

function ContextAwareToggleCol({ children, eventKey, md, callback }) {
  const { activeEventKey } = useContext(AccordionContext);

  const decoratedOnClick = useAccordionButton(
    eventKey,
    () => callback && callback(eventKey),
  );

  const isCurrentEventKey = activeEventKey === eventKey;

  return (
    <Col onClick={decoratedOnClick} md={md}>
      <h5>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-up" viewBox="0 0 16 16">
          <path fill-rule="evenodd" d={isCurrentEventKey ? 
            "M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 0 1-.708-.708l6-6z" :
            "M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"
          }/>
        </svg>
      </h5>
    </Col>
  );
}

function ProcessCard({ children, process_idx, component, title }) {
  const progress_current = component.state.progress_currents[process_idx];
  const progress_total = component.state.progress_totals[process_idx];
  const progress_percent = component.state.progress_percents[process_idx];
  const past_time = component.state.past_times[process_idx];
  const est_time = component.state.estimate_times[process_idx];
  
  return (
    <Card className="my-3">
      <Card.Header className="m-2">
        <Container>
          <Row>
            <Col md="auto">
              <input
                type="checkbox"
                id={'check' + process_idx.toString()}
                checked={component.state.jobs[process_idx]}
                onChange={e => {
                  var new_arr = component.state.jobs.concat();
                  new_arr[process_idx] = e.target.checked;
                  component.setState({ jobs: new_arr });
                }}
              />
            </Col>
            <CustomToggleCol eventKey={process_idx} md="auto">
              <label for={'check' + process_idx.toString()}>
                <h5>{title}</h5>
              </label>
            </CustomToggleCol>
            <Col md="auto">
              <form onSubmit={e => {
                var new_jobs = [false, false, false, false];
                new_jobs[process_idx] = true;
                component.setState({ jobs: new_jobs });
                component.handleSubmitJobs(e, new_jobs);
              }}>
                <input type="submit" value="実行" disabled={component.state.processing !== -1}/>　
                {/* 処理中のぐるぐる（スピナー） */
                  component.state.processing === process_idx &&
                  (progress_current !== progress_total) &&
                  <Spinner animation="border" size="sm" />
                }
              </form>
            </Col>
            <CustomToggleCol eventKey={process_idx}>
              {/* プログレスバー */
                progress_current === 0 ||
                (progress_current !== progress_total) ?
                <ProgressBar
                  className="my-1"
                  animated
                  now={progress_percent}
                  label={`${progress_percent}% (${progress_current}/${progress_total})`}
                /> :
                <ProgressBar
                  className="my-1"
                  variant="success"
                  now={progress_percent}
                  label={`${progress_percent}% (${progress_current}/${progress_total})`}
                />
              }
            </CustomToggleCol>
            <CustomToggleCol eventKey={process_idx} md="auto">{past_time} 残り {est_time}</CustomToggleCol>
            <ContextAwareToggleCol eventKey={process_idx} md="auto"/>
          </Row>
        </Container>
      </Card.Header>
      <Accordion.Collapse eventKey={process_idx}>
        <Card.Body>
          {children}
        </Card.Body>
      </Accordion.Collapse>
    </Card>
  );
}

function timerCountToStr(count) {
  var c = count
  const seconds = c % 60
  c = Math.floor(c / 60)
  const minutes = c % 60
  const hour = Math.floor(c / 60)
  const ms_str = minutes.toString().padStart(2, "0") + ":" + seconds.toString().padStart(2, "0")
  if (hour > 0) {
    return hour.toString() + ":" + ms_str
  } else {
    return ms_str
  }
}

function calcEstRemain(count, current, total) {
  // 1処理の平均時間
  const avg = count / current;
  // 全て終えるのにかかる時間
  const total_time = avg * total;
  // 推定残り時間
  return Math.round(total_time - count)
}

export class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      jobs: [false, false, false, false],
      processing: -1,
      timer_count: 0,
      est_remain: 0,
      progress_currents: [0, 0, 0, 0],
      progress_totals: [0, 0, 0, 0],
      progress_percents: [0, 0, 0, 0],
      past_times: ['--:--', '--:--', '--:--', '--:--'],
      estimate_times: ['--:--', '--:--', '--:--', '--:--'],
      estimate_filter_markets: [],
      selected_estimate_filter_market: '',
      estimate_methods: [],
      selected_estimate_method: '',
      selected_estimate_method_desc: '',
      rebuild_estimate_model: false,
      strategy_methods: [],
      selected_strategy_method: '',
      selected_strategy_method_desc: '',
      evaluate_start: '2016-01-01',
      evaluate_period: 20,
      evaluate_base: 1000000,
      evaluate_gains: 100000,
    };

    this.handleSubmitJobs = this.handleSubmitJobs.bind(this);
  }

  componentDidMount() {
    // ソケット通信で経過時間を受け取った時
    socket.on('timer_count', (data) => {
      const new_remain = Math.max(this.state.est_remain - 1, 0);
      const remain_str = timerCountToStr(new_remain);
      this.setState({
        timer_count: data.count,
        est_remain: new_remain,
      });
      var new_past_times = this.state.past_times.concat();
      var new_estimate_times = this.state.estimate_times.concat();
      const processing = this.state.processing;
      if (0 <= processing && processing < new_past_times.length) {
        new_past_times[processing] = timerCountToStr(data.count);
        new_estimate_times[processing] = remain_str;
        this.setState({
          past_times: new_past_times,
          estimate_times: new_estimate_times,
        });
      }
    });

    // ソケット通信で現在の処理進捗情報を受け取った時
    socket.on('progress', (data) => {
      var percent = 0;
      const processing = this.state.processing;
      if (data.total !== 0) {
        percent = Math.round((data.current / data.total) * 100);
        if (data.current > 0) {
          // 推定残り時間
          const est_remain = calcEstRemain(this.state.timer_count, data.current, data.total);
          var new_estimate_times = this.state.estimate_times.concat();
          if (0 <= processing && processing < new_estimate_times.length) {
            new_estimate_times[processing] = timerCountToStr(est_remain);
          }
          this.setState({
            est_remain: est_remain,
            estimate_times: new_estimate_times,
          });
        }
      }
      var new_progress_currents = this.state.progress_currents.concat();
      var new_progress_totals = this.state.progress_totals.concat();
      var new_progress_percents = this.state.progress_percents.concat();
      if (0 <= processing && processing < new_progress_currents.length) {
        new_progress_currents[processing] = data.current;
        new_progress_totals[processing] = data.total;
        new_progress_percents[processing] = percent;
      }
      this.setState({
        progress_currents: new_progress_currents,
        progress_totals: new_progress_totals,
        progress_percents: new_progress_percents,
      });
      if (data.total !== 0 && data.current === data.total) {
        this.setState({processing: -1});
      } else {
        this.setState({processing: data.processing_job});
      }
    });

    // ソケット通信で市場フィルタ情報を受け取った時
    Axios.get('http://127.0.0.1:8000/estimate/filter_markets').then((response) => {
      this.setState({
        estimate_filter_markets: response.data.markets,
      });
    });

    // ソケット通信で推定方法の情報を受け取った時
    Axios.get('http://127.0.0.1:8000/estimate/methods').then((response) => {
      this.setState({
        estimate_methods: response.data.methods,
      });
    });

    // ソケット通信で戦略方法の情報を受け取った時
    Axios.get('http://127.0.0.1:8000/strategy/methods').then((response) => {
      this.setState({
        strategy_methods: response.data.methods,
      });
    });

    // ソケット通信で各処理の情報を受け取った時
    Axios.get('http://127.0.0.1:8000/get_jobs').then((response) => {
      this.setState({
        jobs: response.data.jobs,
        processing: response.data.processing_job,
        selected_estimate_method: response.data.estimate_method,
        selected_estimate_filter_market: response.data.filter_market,
        rebuild_estimate_model: response.data.estimate_rebuild_model,
        selected_strategy_method: response.data.strategy_method,
        evaluate_start: response.data.evaluate_start,
        evaluate_period: response.data.evaluate_period,
        evaluate_base: response.data.evaluate_base,
        evaluate_gains: response.data.evaluate_gains,
      });
      const processing = response.data.processing_job;
      var percent = 0;
      if (response.data.progress_total !== 0) {
        percent = Math.round((response.data.progress_current / response.data.progress_total) * 100);
        if (response.data.current > 0) {
          // 推定残り時間
          const est_remain = calcEstRemain(response.data.timer_count, response.data.current, response.data.total);
          var new_estimate_times = this.state.estimate_times.concat();
          if (0 <= processing && processing < new_estimate_times.length) {
            new_estimate_times[processing] = timerCountToStr(est_remain);
          }
          this.setState({
            est_remain: est_remain,
            estimate_times: new_estimate_times,
          });
        }
      }
      var new_progress_currents = this.state.progress_currents.concat();
      var new_progress_totals = this.state.progress_totals.concat();
      var new_progress_percents = this.state.progress_percents.concat();
      if (0 <= processing && processing < new_progress_currents.length) {
        new_progress_currents[processing] = response.data.progress_current;
        new_progress_totals[processing] = response.data.progress_total;
        new_progress_percents[processing] = percent;
      }
      this.setState({
        progress_currents: new_progress_currents,
        progress_totals: new_progress_totals,
        progress_percents: new_progress_percents,
      });
    });
  }

  componentWillUnmount() {
    socket.off('timer_count');
    socket.off('progress');
  }

  render() {
    return (
      <Container>
          <h1>Investment Support with AI</h1>
          <form onSubmit={e => {this.handleSubmitJobs(e, this.state.jobs)}}>
            <input type="submit" value="選択した処理を上から順に実行" disabled={this.state.processing !== -1}/>
          </form>
          <Accordion alwaysOpen>
            <ProcessCard
              process_idx={0}
              component={this}
              title="銘柄データ取得"
            />

            <ProcessCard
              process_idx={1}
              component={this}
              title="株価予想"
            >
              <h6>オプション</h6>
                <form>
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
                        onChange={e => {
                          this.setState({ selected_estimate_method: e.target.value });
                          for (const method of this.state.estimate_methods) {
                            if (method.name == e.target.value) {
                              this.setState({ selected_estimate_method_desc: method.description });
                              break;
                            }
                          }
                        }}
                      >
                        {this.state.estimate_methods.map((method) => {
                          return <option value={method.name}>{method.name}</option>
                        })}
                      </select>
                      　{this.state.selected_estimate_method_desc}
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
                </form>
            </ProcessCard>

            <ProcessCard
              process_idx={2}
              component={this}
              title="戦略"
            >
              <h6>オプション</h6>
                <form>
                  <p>
                    <label>
                      戦略：
                      <select
                        value={this.state.selected_strategy_method}
                        onChange={e => {
                          this.setState({ selected_strategy_method: e.target.value });
                          for (const method of this.state.strategy_methods) {
                            if (method.name == e.target.value) {
                              this.setState({ selected_strategy_method_desc: method.description });
                              break;
                            }
                          }
                        }}
                      >
                        {this.state.strategy_methods.map((method) => {
                          return <option value={method.name}>{method.name}</option>
                        })}
                      </select>
                      　{this.state.selected_strategy_method_desc}
                    </label>
                  </p>
                </form>
            </ProcessCard>

            <ProcessCard
              process_idx={3}
              component={this}
              title="評価"
            >
              <h6>オプション</h6>
                <form>
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
                </form>
            </ProcessCard>
          </Accordion>
      </Container>
    );
  }


  handleSubmitJobs = (event, jobs) => {
    var formData = new FormData();
    formData.append('jobs', jobs);
    formData.append('estimate_method', this.state.selected_estimate_method);
    formData.append('filter_market', this.state.selected_estimate_filter_market);
    formData.append('estimate_rebuild_model', this.state.rebuild_estimate_model);
    formData.append('strategy_method', this.state.selected_strategy_method);
    formData.append('evaluate_base', this.state.evaluate_base);
    formData.append('evaluate_start', this.state.evaluate_start);
    formData.append('evaluate_period', this.state.evaluate_period);
    formData.append('evaluate_gains', this.state.evaluate_gains);
    const customHeader = {
      headers: {
        "Content-Type": 'multipart/form-data',
      },
    };
    Axios.post('http://127.0.0.1:8000/request_jobs', formData, customHeader)
    .then(function(res) {
    })
    // ブラウザによるページの更新を阻止
    event.preventDefault();
  };
}

export default App;
