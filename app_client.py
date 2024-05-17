from crypt import methods
from time import sleep
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
        "Message": "app up and running successfully"
    })



@app.route("/access",methods=["POST"])
def access():
    data = request.get_json()
    print(data)
    name = data.get("name", "dipto")
    server = data.get("server","server1")

    message = f"User {name} received access to server {server}"

    return jsonify({
        "Message": message
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
    world_size = data.get("world_size", 3)
    rank = data.get("rank")
    model_name = data.get("model_name", "alexnet")
    dataset_name = data.get("dataset_name", [])
    epochs = data.get("epochs")
    lr = data.get("lr", 0.001)
    dataset_id = data.get("dataset_id")
    batch_size = data.get("batch_size", 32)
    optim = data.get("optim")

    # Executar fl_client.py em uma thread separada
    thread = threading.Thread(target=run_fl_client, args=(
    server_ip, port, world_size, rank, model_name, dataset_name, epochs, lr, dataset_id, batch_size, optim))
    thread.start()

    message = f"Federated Learning Process started on client {rank}."

    sleep()
    return jsonify({"Message": message})


@app.route('/server-status', methods=['GET'])
def check_server():
    if process and process.poll() is None:
        return jsonify({"status": "The server is running a training."}), 503
    else:
        return jsonify({"status": "The server is twiddling its thumbs, waiting for action."}), 200

def run_fl_client(server_ip, port, world_size, rank, model_name, dataset_name, epochs, lr, dataset_id, batch_size, optim):
    global process 
    process = subprocess.Popen(["python3", "scripts/fl_client.py",
                    "--server_ip", server_ip,
                    "--port", port,
                    "--world_size", str(world_size),
                    "--rank", str(rank),
                    "--model_name", model_name,
                    "--dataset_name"] + dataset_name +
                   ["--epochs", str(epochs),
                    "--lr", str(lr),
                    "--dataset_id", str(dataset_id),
                    "--batch_size", str(batch_size),
                    "--optim", optim])


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8081)