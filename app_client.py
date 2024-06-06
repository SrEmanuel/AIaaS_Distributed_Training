from crypt import methods
from time import sleep
from flask import Flask,jsonify,request
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import subprocess

from constants import endpoints, arguments_position
from helpers import auxiliary

process:list = []

app = Flask(__name__)

swagger_ui_blueprint = get_swaggerui_blueprint(
    endpoints.SWAGGER_URL,
    endpoints.API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=endpoints.SWAGGER_URL)

@app.route("/")
def home():
    return jsonify({
        "Message": "app up and running successfully"
    })

@app.route("/access",methods=["POST"])
def access():
    data = request.get_json()
    
    if data is None:
        print('[ERROR] No body found for access request!')
        return

    name = data.get("name", "dipto")
    server = data.get("server","server1")

    message = f"User {name} received access to server {server}"

    return jsonify({
        "Message": message
    })

@app.route("/run/train", methods=["POST"])
def train():
    data = request.get_json()
    print('[INFO] JSON data for request', data)

    if not data:
        print('[ERROR] No body found for request!')
        return
    
    port = data.get('port', "8081")
    print('[INFO] getting training PORT', port)

    global process
    hasTrainingInPort = auxiliary.contains(
        process, 
        lambda x: x.args[arguments_position.TRAINING_PORT_ARG_POSITION] == port
        )

    if hasTrainingInPort:
        return jsonify({"status": "This server is already running a training. Please, wait for it become free to go again"}), 503
    
    print("[INFO] Process Staring...")

    server_ip = data.get("server_ip", "127.0.0.1")
    port = data.get("port", "8081")
    world_size = data.get("worldSize", 3)
    rank = data.get("rank")
    model_name = data.get("model_name", "alexnet")
    dataset_name = data.get("datasetName", [])
    epochs = data.get("epochs")
    lr = data.get("lr", 0.001)
    dataset_id = data.get("datasetId", 12)
    batch_size = data.get("batch_size", 32)
    optim = data.get("optim")
    sleep(5)

    # Executar fl_client.py em uma thread separada
    thread = threading.Thread(target=run_fl_client, args=(
    server_ip, port, world_size, rank, model_name, dataset_name, epochs, lr, dataset_id, batch_size, optim))
    thread.start()

    message = f"Federated Learning Process started on client {rank}."

    return jsonify({"Message": message})


@app.route('/server-status', methods=['GET'])
def check_server():
    return jsonify({"status": "Server is up and running"})

def run_fl_client(server_ip, port, world_size, rank, model_name, dataset_name, epochs, lr, dataset_id, batch_size, optim):
    localProcess = subprocess.Popen(["python3", "scripts/fl_client.py",
                    "--server_ip", str(server_ip),
                    "--port", str(port),
                    "--world_size", str(world_size),
                    "--rank", str(rank),
                    "--model_name", model_name,
                    "--epochs", str(epochs),
                    "--lr", str(lr),
                    "--dataset_id", str(dataset_id),
                    "--batch_size", str(batch_size),
    	                "--optim", optim])
    
    global process 
    process.append(localProcess)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
