from flask import Flask, request, jsonify, abort
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../fetch'))
import fetch_data as fetch_data
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import estimate.estimate_1 as estimate_1
import estimate.estimate_lstm_2 as estimate_lstm_2
import strategy.strategy_1 as strategy_1

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

@app.route('/fetch', methods=['POST'])
def fetch():
    socketio.start_background_task(target=fetch_task)
    return "fetch started", 202

def fetch_task():
    for i in fetch_data.fetch_data_gen('', '', []):
        progress = round((i[0] + 1) / i[1] * 100)
        socketio.emit('progress', {'progress': progress})

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
    filter_market_code = 0
    if filter_market in market_to_code:
        filter_market_code = market_to_code[filter_market]
    socketio.start_background_task(target=estimate_task(method=method, filter_market_code=filter_market_code))
    return "estimate started", 202
    
@app.route('/estimate/methods', methods=['GET'])
def get_estimate_methods():
    data = {
        'methods': ['SMA', 'LSTM2'],
    }
    return jsonify(data)

@app.route('/estimate/filter_markets', methods=['GET'])
def get_estimate_filter_markets():
    data = {
        'markets': list(market_to_code.keys()),
    }
    return jsonify(data)

def estimate_task(method: str, filter_market_code = 0):
    if method == 'SMA':
        for i in estimate_1.estimate_gen('', ''):
            progress = round((i[0] + 1) / i[1] * 100)
            socketio.emit('progress_estimate', {'progress': progress})
    elif method == 'LSTM2':
        for i in estimate_lstm_2.estimate_gen('', '', filter_market_code):
            progress = round((i[0] + 1) / i[1] * 100)
            socketio.emit('progress_estimate', {'progress': progress})

@app.route('/strategy', methods=['POST'])
def strategy():
    method = request.form['method']
    filter_market = request.form['filter_market']
    filter_market_code = 0
    if filter_market in market_to_code:
        filter_market_code = market_to_code[filter_market]
    socketio.start_background_task(target=strategy_task(method=method, filter_market_code=filter_market_code))
    return "strategy started", 202

@app.route('/strategy/methods', methods=['GET'])
def get_strategy_methods():
    data = {
        'methods': ['strategy1'],
    }
    return jsonify(data)

def strategy_task(method: str, filter_market_code = 0):
    if method == 'strategy1':
        for i in strategy_1.estimate_gen('', ''):
            progress = round((i[0] + 1) / i[1] * 100)
            socketio.emit('progress_strategy', {'progress': progress})

if __name__ == "__main__":
    socketio.run(app, port=8000, debug=True)