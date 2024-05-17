from crypt import methods
from time import sleep
import json
from flask import Flask,jsonify,request
import requests
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import subprocess

process = None
trainingUuid = None
process = None
app = Flask(__name__)
SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"
#O endereço ip no endpoint local se refere ao ip designado para a máquina do IAAS via VPN
IAAS_ENDPOINT_LOCAL="http://10.255.0.2:8080/api/v1/training"
IAAS_ENDPOINT="http://api.iaas.emanuelm.dev/api/v1/training"
actualTrainingUuid= None


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

    global trainingUuid
    trainingUuid = data.get("trainingUuid", "None")
    if(trainingUuid == None):
        return
    
    numrounds = data.get("rounds", 3)
    server_ip = data.get("server_ip", "127.0.0.1")
    port = data.get("port", "3002")
    model_name = data.get("model_name", "alexnet")
    
    # Executar fl_client.py em uma thread separada
    thread = threading.Thread(target=run_fl_server, args=(
    server_ip, port, model_name, trainingUuid, numrounds))
    thread.start()

    message = f"Federated Learning Server started successfully."

    return jsonify({"Message": message})


@app.route('/server-status', methods=['GET'])
def check_server():
    if process and process.poll() is None:
        return jsonify({"status": "The server is running a training."}), 503
    else:
        return jsonify({"status": "The server is twiddling its thumbs, waiting for action."}), 200
 
def performTrainingRequest():
    
    file = open("./scripts/teste.pdf", 'rb')
    
    training_finish_dto = {
        "timestamp": "2024-05-15T10:00:00Z",
        "status": "success",
        "data": "Training data",
        "errors": []
    }
    
    #Alterar o nome aqui em baixo para ele buscar o .pth ao invés do .pdf
    parts = {
        'file': ('teste.pdf', file),
        'request': ('json', json.dumps(training_finish_dto), 'application/json')
    }
    
    global trainingUuid
    requests.post(IAAS_ENDPOINT_LOCAL+"/finishTraining/"+trainingUuid, files=parts)
    
def checkTrainingFinish():
    global process
    while(process is not None and process.poll() is None):
        print("-> o treinamento ainda não terminou!")
        sleep(10)

    print("O treinamento terminou!")
    performTrainingRequest()
   
def run_fl_server(server_ip, port, model_name, trainingUuid, rounds):
    global process 
    process = subprocess.Popen(["python3", "scripts/fl_server.py",
                    "--port", str(port),
                    "--model_name", str(model_name),
                    "--trainingUuid", str(trainingUuid),
                    "--numRounds", str(rounds)])
    checkTrainingFinish()


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8081)
