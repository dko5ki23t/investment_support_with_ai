from flask import Flask, request, render_template
import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../fetch'))
import fetch_data as fetch_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fetch_data.fetch_data('', '', [])
        return render_template('fetch_data.html')
    else:
        return render_template('fetch_data.html')

if __name__ == "__main__":
    app.run(port=8000, debug=True)