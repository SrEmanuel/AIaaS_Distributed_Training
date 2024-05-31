from crypt import methods
from time import sleep
import json
from flask import Flask,jsonify,request
import requests
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import subprocess

from helpers import auxiliary
from constants import endpoints, arguments_position

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
        "Message": "Server's instance is app up and running successfully"
    })
    
@app.route("/run/train", methods=["POST"])
def train():
    data = request.get_json()
    print('[INFO] JSON data from request', data)

    port = data.get("port", "3002")
    print('[INFO] getting training PORT', port)

    global process
    hasTrainingInPort = auxiliary.contains(process, lambda x: x.args[arguments_position.TRAINING_PORT_ARG_POSITION] == port)

    if hasTrainingInPort:
        return jsonify({"status": "[INFO] This server is already running a training. Please, wait for it become free to go again"}), 503
    
    print("[INFO] Process staring...")

    currentTrainingUuid = data.get("trainingUuid", None)

    if(currentTrainingUuid == None):
        print('[ERROR] trainingUuid not found on request body')
        return

    numrounds = data.get("rounds", 3)
    server_ip = data.get("server_ip", "127.0.0.1")
    model_name = data.get("model_name", "alexnet")
    
    # Executar fl_client.py em uma thread separada
    thread = threading.Thread(target=run_fl_server, args=(
    server_ip, port, model_name, currentTrainingUuid, numrounds))
    thread.start()

    message = f"[INFO] federated Learning Server started successfully."

    return jsonify({"Message": message})


@app.route('/server-status', methods=['GET'])
def check_server():
    desirablePort = request.args.get('port')

    if not desirablePort:
        print('[INFO] The desirable port was not found in query params.')
        return jsonify({"status": "The port was not found on queryString. Please, dispose the desirable port as a query param"}), 500  

    global process
    localProcess = auxiliary.contains(process, lambda x: x.args[arguments_position.TRAINING_PORT_ARG_POSITION] == desirablePort, False)

    if not localProcess:
        print('[INFO] Process not found in queue')
        return jsonify({"status": "The desired process was not found in queue. Please try with other PORT"}), 500  

    if localProcess and localProcess.poll() is None:
        return jsonify({"status": "The server is running a training."}), 503
    
    return jsonify({"status": "The server is twiddling its thumbs, waiting for action."}), 200  

def performTrainingRequest(port):
    file = open("./scripts/teste.pdf", 'rb')
    
    training_finish_dto = {
        "timestamp": "2024-05-15T10:00:00Z",
        "status": "success",
        "data": "Training datraining_finish_dtota",
        "errors": []
    }
    
    #Alterar o nome aqui em baixo para ele buscar o .pth ao invés do .pdf
    parts = {
        'file': ('teste.pdf', file),
        'request': ('json', json.dumps(training_finish_dto), 'application/json')
    }
    
    global process
    currentProcess = auxiliary.contains(process, lambda x: x.args[arguments_position.TRAINING_PORT_ARG_POSITION] == port, False)
    currentUuid = currentProcess.args[arguments_position.TRAINING_UUID_ARG_POSITION]
    
    requests.post(endpoints.IAAS_ENDPOINT_LOCAL+"/finishTraining/"+currentUuid, files=parts)

def checkTrainingFinish(port):
    global process
    localProcess = auxiliary.contains(process, lambda x: x.args[arguments_position.TRAINING_PORT_ARG_POSITION] == port, False)

    if not localProcess:
        print('[ERROR] process not found while checking if has finished')
        return
    
    while(localProcess is not None and localProcess.poll() is None):
        print("[INFO - training ", port, "] O treinamento da porta", port, "ainda não terminou!")
        sleep(10)

    print("[INFO - training ", port, "] O treinamento da porta ", port, " terminou!")
    performTrainingRequest(port)
   
def run_fl_server(server_ip, port, model_name, trainingUuid, rounds):
    localProcess = subprocess.Popen(["python3", "scripts/fl_server.py",
                    "--port", str(port),
                    "--model_name", str(model_name),
                    "--trainingUuid", str(trainingUuid),
                    "--numRounds", str(rounds)])
        
    global process 
    process.append(localProcess)
    checkTrainingFinish(port)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8081)
