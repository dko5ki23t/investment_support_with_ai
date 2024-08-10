from flask import Flask, request, jsonify, abort
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../fetch'))
import fetch_data as fetch_data
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import estimate.estimate_sma as estimate_sma
import estimate.estimate_lstm as estimate_lstm_1
import estimate.estimate_lstm_2 as estimate_lstm_2
import estimate.estimate_lstm_3 as estimate_lstm_3
import estimate.estimate_lstm_4 as estimate_lstm_4
import estimate.estimate_lstm_5 as estimate_lstm_5
import strategy.strategy_1 as strategy_1
import strategy.strategy_2 as strategy_2
import strategy.strategy_3 as strategy_3
import strategy.strategy_4 as strategy_4
import strategy.strategy_5 as strategy_5
import strategy.strategy_6 as strategy_6
import strategy.strategy_7 as strategy_7
import strategy.strategy_8 as strategy_8
import evaluate.evaluate as evaluate

app = Flask(__name__)
# 日本語を使えるようにする
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app) #Cross Origin Resource Sharing

market_to_code = {
    'プライム': 111,
    'スタンダード': 112,
    'グロース': 113,
}

estimate_default_root = os.path.join(os.path.dirname(__file__), '../../db/estimates/')
estimate_method_to_module = {
    'SMA': estimate_sma,
    'LSTM1': estimate_lstm_1,
    'LSTM2': estimate_lstm_2,
    'LSTM3': estimate_lstm_3,
    'LSTM4': estimate_lstm_4,
    'LSTM5': estimate_lstm_5,
}
estimate_method_to_directory = {
    'SMA': os.path.join(estimate_default_root, 'estimate_sma'),
    'LSTM1': os.path.join(estimate_default_root, 'estimate_lstm_1'),
    'LSTM2': os.path.join(estimate_default_root, 'estimate_lstm_2'),
    'LSTM3': os.path.join(estimate_default_root, 'estimate_lstm_3'),
    'LSTM4': os.path.join(estimate_default_root, 'estimate_lstm_4'),
    'LSTM5': os.path.join(estimate_default_root, 'estimate_lstm_5'),
}

strategy_default_root = os.path.join(os.path.dirname(__file__), '../../db/orders/')
strategy_method_to_module = {
    'strategy1': strategy_1,
    'strategy2': strategy_2,
    'strategy3': strategy_3,
    'strategy4': strategy_4,
    'strategy5': strategy_5,
    'strategy6': strategy_6,
    'strategy7': strategy_7,
    'strategy8': strategy_8,
}
strategy_method_to_directory = {
    'strategy1': os.path.join(strategy_default_root, 'order_1'),
    'strategy2': os.path.join(strategy_default_root, 'order_2'),
    'strategy3': os.path.join(strategy_default_root, 'order_3'),
    'strategy4': os.path.join(strategy_default_root, 'order_4'),
    'strategy5': os.path.join(strategy_default_root, 'order_5'),
    'strategy6': os.path.join(strategy_default_root, 'order_6'),
    'strategy7': os.path.join(strategy_default_root, 'order_7'),
    'strategy8': os.path.join(strategy_default_root, 'order_8'),
}

@app.route('/fetch', methods=['POST'])
def fetch():
    socketio.start_background_task(target=fetch_task)
    return "fetch started", 202

def fetch_task():
    for i in fetch_data.fetch_data_gen('', '', []):
        socketio.emit('progress_fetch', {'current': i[0] + 1, 'total': i[1]})

@app.route('/estimate', methods=['POST'])
def estimate():
#    if request.method == 'GET':
#        data = {'methods': ['SMA', 'LSTM2']}
#        print('OK')
#        return jsonify(data)
#    elif request.method == 'POST':
#        print('OK2')
#        method = request.form['method']
#        socketio.start_background_task(target=estimate_task(method=method))
#        return "estimate started", 202
#    else:
#        return abort(400)
    method = request.form['method']
    filter_market = request.form['filter_market']
    rebuild_model = (request.form['rebuild_model']).lower() == 'true'
    filter_market_code = 0
    if filter_market in market_to_code:
        filter_market_code = market_to_code[filter_market]
    socketio.start_background_task(
        target=estimate_task(
            method=method,
            filter_market_code=filter_market_code,
            rebuild_model=rebuild_model
        ))
    return "estimate started", 202
    
@app.route('/estimate/methods', methods=['GET'])
def get_estimate_methods():
    data = {
        'methods': list(estimate_method_to_directory.keys()),
    }
    return jsonify(data)

@app.route('/estimate/filter_markets', methods=['GET'])
def get_estimate_filter_markets():
    data = {
        'markets': list(market_to_code.keys()),
    }
    return jsonify(data)

def estimate_task(method: str, filter_market_code = 0, rebuild_model = False):
    for i in estimate_method_to_module[method].estimate_gen('', '', filter_market_code=filter_market_code, force_build_model=rebuild_model):
        socketio.emit('progress_estimate', {'current': i[0] + 1, 'total': i[1]})

@app.route('/strategy', methods=['POST'])
def strategy():
    estimate_method = request.form['estimate_method']
    strategy_method = request.form['strategy_method']
    base = int(request.form['base'])
    filter_market = request.form['filter_market']
    filter_market_code = 0
    if filter_market in market_to_code:
        filter_market_code = market_to_code[filter_market]
    socketio.start_background_task(
        target=strategy_task(
            estimate_method=estimate_method,
            strategy_method=strategy_method,
            base=base,
            filter_market_code=filter_market_code
        )
    )
    return "strategy started", 202

@app.route('/strategy/methods', methods=['GET'])
def get_strategy_methods():
    data = {
        'methods': list(strategy_method_to_directory.keys()),
    }
    return jsonify(data)

def strategy_task(estimate_method: str, strategy_method: str, base: int, filter_market_code = 0):
    input = estimate_method_to_directory[estimate_method]
    for i in strategy_method_to_module[strategy_method].strategy_gen(input, base=base, filter_market_code=filter_market_code, method_name=estimate_method):
        socketio.emit('progress_strategy', {'current': i[0] + 1, 'total': i[1]})

@app.route('/evaluate', methods=['POST'])
def evaluate_app():
    estimate_method = request.form['estimate_method']
    strategy_method = request.form['strategy_method']
    start = request.form['start']
    period = int(request.form['period'])
    base = int(request.form['base'])
    gains = int(request.form['gains'])
    filter_market = request.form['filter_market']
    filter_market_code = 0
    if filter_market in market_to_code:
        filter_market_code = market_to_code[filter_market]
    socketio.start_background_task(
        target=evaluate_task(
            estimate_method, strategy_method,
            start, period, base, gains,
            filter_market_code,
        )
    )
    return "evaluate started", 202

def evaluate_task(
        estimate_method: str,
        strategy_method: str,
        start: str, period: int,
        base: int, gains: int,
        filter_market_code=0,
):
    input = os.path.join(
        strategy_method_to_directory[strategy_method],
        f'order_{filter_market_code}_{estimate_method}.json')
    for i in evaluate.evaluate_gen(input, start=start, period=period, base=base, gains=gains):
            socketio.emit('progress_evaluate', {'current': i[0] + 1, 'total': i[1]})

if __name__ == "__main__":
    socketio.run(app, port=8000, debug=True)