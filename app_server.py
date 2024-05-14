from crypt import methods
from flask import Flask,jsonify,request
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import subprocess

process = None

app = Flask(__name__)
SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

@app.route("/")
def home():
    return jsonify({
        "Message": "Server's instance is app up and running successfully"
    })
    
@app.route("/run/train", methods=["POST"])
def train():
    if process and process.poll() is None:
        return jsonify({"status": "This server is already running a training. Please, wait for it become free to go again"}), 503
    
    print("Staring...")
    data = request.get_json()
    print(data)

    server_ip = data.get("server_ip", "127.0.0.1")
    port = data.get("port", "3002")
    model_name = data.get("model_name", "alexnet")
    # Executar fl_client.py em uma thread separada
    thread = threading.Thread(target=run_fl_server, args=(
    server_ip, port, model_name,))
    thread.start()

    message = f"Federated Learning Server started successfully."

    return jsonify({"Message": message})

@app.route('/server-status', methods=['GET'])
def check_server():
    if process and process.poll() is None:
        return jsonify({"status": "The server is running a training."}), 503
    else:
        return jsonify({"status": "The server is twiddling its thumbs, waiting for action."}), 200

def run_fl_server(server_ip, port, model_name):
    global process 
    process = subprocess.Popen(["python3", "scripts/fl_server.py",
                    "--port", port,
                    "--model_name", model_name])


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)