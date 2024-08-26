from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import sys
import os
import threading
from logging import NullHandler

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fetch.fetch_data as fetch_data
import estimate.estimate_sma as estimate_sma
import estimate.estimate_lstm as estimate_lstm_1
import estimate.estimate_lstm_2 as estimate_lstm_2
import estimate.estimate_lstm_3 as estimate_lstm_3
import estimate.estimate_lstm_4 as estimate_lstm_4
import estimate.estimate_lstm_5 as estimate_lstm_5
import estimate.estimate_lstm_6 as estimate_lstm_6
import estimate.estimate_lstm_7 as estimate_lstm_7
import estimate.estimate_lstm_8 as estimate_lstm_8
import strategy.strategy_1 as strategy_1
import strategy.strategy_2 as strategy_2
import strategy.strategy_3 as strategy_3
import strategy.strategy_4 as strategy_4
import strategy.strategy_5 as strategy_5
import strategy.strategy_6 as strategy_6
import strategy.strategy_7 as strategy_7
import strategy.strategy_8 as strategy_8
import strategy.strategy_9 as strategy_9
import strategy.strategy_10 as strategy_10
import evaluate.evaluate as evaluate

app = Flask(__name__)
# 日本語を使えるようにする
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = 'secret!'
# ソケット通信準備
socketio = SocketIO(app, cors_allowed_origins="*")
# Cross Origin Resource Sharing 
CORS(app)

# タイマー停止フラグ
stop_timer_flag = True
# タイマー
tm = NullHandler
# タイマーカウント（秒）
timer_count = 0

# タイマーで1秒ごとに呼び出されるコールバック
def tm_callback():
    global tm, timer_count
    tm.cancel()
    del tm
    tm = NullHandler
    timer_count = timer_count + 1
    # タイマーカウントをソケット通信で送る
    socketio.emit('timer_count', {'count': timer_count})
    # 次の1秒タイマーをセット
    if not stop_timer_flag:
        tm = threading.Timer(1, tm_callback)
        tm.start()

# タイマーを開始する
def start_timer():
    global tm, timer_count, stop_timer_flag
    timer_count = 0
    stop_timer_flag = False
    tm = threading.Timer(1, tm_callback)
    tm.start()

# タイマーを停止する
def stop_timer():
    global tm, stop_timer_flag
    stop_timer_flag = True
    tm.cancel()
    tm = NullHandler

# 行う処理リスト（fetch, estimate, strategy, evaluateの処理有無）
jobs = [False, False, False, False]
# 現在進行中の処理インデックス（-1は現在進行中の処理なし）
processing_job = -1
# 進行中処理の進捗
progress_current = 0
progress_total = 0
# 各処理で用いるパラメータ
params = {
    'estimate_method': 'SMA',
    'filter_market': 'プライム',
    'estimate_rebuild_model': False,
    'strategy_method': 'strategy1',
    'evaluate_base': 1000000,
    'evaluate_start': '2016-01-01',
    'evaluate_period': 20,
    'evaluate_gains': 100000,
}
# 市場名->市場コード
market_to_code = {
    'プライム': 111,
    'スタンダード': 112,
    'グロース': 113,
}
# 推定結果を格納するフォルダ名
estimate_default_root = os.path.join(os.path.dirname(__file__), '../../db/estimates/')
# 推定方法名->使用するモジュール
estimate_method_to_module = {
    'SMA': estimate_sma,
    'LSTM1': estimate_lstm_1,
    'LSTM2': estimate_lstm_2,
    'LSTM3': estimate_lstm_3,
    'LSTM4': estimate_lstm_4,
    'LSTM5': estimate_lstm_5,
    'LSTM6': estimate_lstm_6,
    'LSTM7': estimate_lstm_7,
    'LSTM8': estimate_lstm_8,
}
# 推定方法名->推定結果が格納されたフォルダ名
estimate_method_to_directory = {
    'SMA': os.path.join(estimate_default_root, 'estimate_sma'),
    'LSTM1': os.path.join(estimate_default_root, 'estimate_lstm_1'),
    'LSTM2': os.path.join(estimate_default_root, 'estimate_lstm_2'),
    'LSTM3': os.path.join(estimate_default_root, 'estimate_lstm_3'),
    'LSTM4': os.path.join(estimate_default_root, 'estimate_lstm_4'),
    'LSTM5': os.path.join(estimate_default_root, 'estimate_lstm_5'),
    'LSTM6': os.path.join(estimate_default_root, 'estimate_lstm_6'),
    'LSTM7': os.path.join(estimate_default_root, 'estimate_lstm_7'),
    'LSTM8': os.path.join(estimate_default_root, 'estimate_lstm_8'),
}
# 注文指示を格納するフォルダ名
strategy_default_root = os.path.join(os.path.dirname(__file__), '../../db/orders/')
# 戦略名->使用するモジュール
strategy_method_to_module = {
    'strategy1': strategy_1,
    'strategy2': strategy_2,
    'strategy3': strategy_3,
    'strategy4': strategy_4,
    'strategy5': strategy_5,
    'strategy6': strategy_6,
    'strategy7': strategy_7,
    'strategy8': strategy_8,
    'strategy9': strategy_9,
    'strategy10': strategy_10,
}
# 戦略名->注文指示が格納されたフォルダ名
strategy_method_to_directory = {
    'strategy1': os.path.join(strategy_default_root, 'order_1'),
    'strategy2': os.path.join(strategy_default_root, 'order_2'),
    'strategy3': os.path.join(strategy_default_root, 'order_3'),
    'strategy4': os.path.join(strategy_default_root, 'order_4'),
    'strategy5': os.path.join(strategy_default_root, 'order_5'),
    'strategy6': os.path.join(strategy_default_root, 'order_6'),
    'strategy7': os.path.join(strategy_default_root, 'order_7'),
    'strategy8': os.path.join(strategy_default_root, 'order_8'),
    'strategy9': os.path.join(strategy_default_root, 'order_9'),
    'strategy10': os.path.join(strategy_default_root, 'order_10'),
}

# クライアントからの要求を受け、各処理を実行する
@app.route('/request_jobs', methods=['POST'])
def request_jobs():
    global jobs, params
    # 各パラメータを更新
    jobs_strs = request.form['jobs'].split(',')
    for i in range(min(len(jobs_strs), len(jobs))):
        jobs[i] = jobs_strs[i].lower() == 'true'
    params['estimate_method'] = request.form['estimate_method']
    params['filter_market'] = request.form['filter_market']
    params['estimate_rebuild_model'] = (request.form['estimate_rebuild_model']).lower() == 'true'
    params['strategy_method'] = request.form['strategy_method']
    params['evaluate_base'] = int(request.form['evaluate_base'])
    params['evaluate_start'] = request.form['evaluate_start']
    params['evaluate_period'] = int(request.form['evaluate_period'])
    params['evaluate_gains'] = int(request.form['evaluate_gains'])
    # 各処理をバックグラウンド実行
    socketio.start_background_task(
        target=process_jobs())
    return "process started", 202

# 各処理を実行する
def process_jobs():
    global jobs, params, processing_job, progress_current, progress_total
    filter_market_code = 0
    if params['filter_market'] in market_to_code:
        filter_market_code = market_to_code[params['filter_market']]
    processing_job = -1
    # fetch
    if jobs[0]:
        processing_job = 0
        progress_current = 0
        progress_total = 0
        start_timer()
        for i in fetch_data.fetch_data_gen('', '', []):
            progress_current = i[0] + 1
            progress_total = i[1]
            socketio.emit('progress', {'processing_job': 0, 'current': i[0] + 1, 'total': i[1]})
        stop_timer()
    processing_job = -1
    # estimate
    if jobs[1]:
        processing_job = 1
        progress_current = 0
        progress_total = 0
        start_timer()
        for i in estimate_method_to_module[params['estimate_method']].estimate_gen('', '', filter_market_code=filter_market_code, force_build_model=params['estimate_rebuild_model']):
            progress_current = i[0] + 1
            progress_total = i[1]
            socketio.emit('progress', {'processing_job': 1, 'current': i[0] + 1, 'total': i[1]})
        stop_timer()
    # strategy
    processing_job = -1
    if jobs[2]:
        processing_job = 2
        progress_current = 0
        progress_total = 0
        input = estimate_method_to_directory[params['estimate_method']]
        start_timer()
        for i in strategy_method_to_module[params['strategy_method']].strategy_gen(input, base=params['evaluate_base'], filter_market_code=filter_market_code, method_name=params['estimate_method']):
            progress_current = i[0] + 1
            progress_total = i[1]
            socketio.emit('progress', {'processing_job': 2, 'current': i[0] + 1, 'total': i[1]})
        stop_timer()
    processing_job = -1
    # evaluate
    if jobs[3]:
        processing_job = 3
        progress_current = 0
        progress_total = 0
        input = os.path.join(
            strategy_method_to_directory[params['strategy_method']],
            f'order_{filter_market_code}_{params["estimate_method"]}.json')
        start_timer()
        for i in evaluate.evaluate_gen(input, start=params['evaluate_start'], period=params['evaluate_period'], base=params['evaluate_base'], gains=params['evaluate_gains']):
                progress_current = i[0] + 1
                progress_total = i[1]
                socketio.emit('progress', {'processing_job': 3, 'current': i[0] + 1, 'total': i[1]})
        stop_timer()
    processing_job = -1

# クライアントからの要求を受け、現在進行中の処理の情報等をクライアントに送る
@app.route('/get_jobs', methods=['GET'])
def get_jobs():
    global jobs, timer_count, params, processing_job, progress_current, progress_total
    data = {
        'jobs': jobs,
        'timer_count': timer_count,
        'processing_job': processing_job,
        'progress_current': progress_current,
        'progress_total': progress_total,
        'estimate_method': params['estimate_method'],
        'filter_market': params['filter_market'],
        'estimate_rebuild_model': params['estimate_rebuild_model'],
        'strategy_method': params['strategy_method'],
        'evaluate_base': params['evaluate_base'],
        'evaluate_start': params['evaluate_start'],
        'evaluate_period': params['evaluate_period'],
        'evaluate_gains': params['evaluate_gains'],
    }
    return jsonify(data)

# クライアントからの要求を受け、推定方法についての情報をクライアントに送る
@app.route('/estimate/methods', methods=['GET'])
def get_estimate_methods():
    methods = []
    for k, v in estimate_method_to_module.items():
        methods.append({'name': k, 'description': v.description()})
    data = {
        'methods': methods,
    }
    return jsonify(data)

# クライアントからの要求を受け、フィルタリング可能な市場についての情報をクライアントに送る
@app.route('/estimate/filter_markets', methods=['GET'])
def get_estimate_filter_markets():
    data = {
        'markets': list(market_to_code.keys()),
    }
    return jsonify(data)

# クライアントからの要求を受け、戦略についての情報をクライアントに送る
@app.route('/strategy/methods', methods=['GET'])
def get_strategy_methods():
    methods = []
    for k, v in strategy_method_to_module.items():
        methods.append({'name': k, 'description': v.description()})
    data = {
        'methods': methods,
    }
    return jsonify(data)

if __name__ == "__main__":
    socketio.run(app, port=8000, debug=True)