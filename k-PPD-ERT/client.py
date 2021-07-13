import socket
from src import generate_parties
import tools
import pickle
import numpy as np

def receive_message(_client_socket):
    try:
        message_header = _client_socket.recv(HEADER_LENGTH)
        if not len(message_header):
            return False
        message_length = int(message_header.decode('utf-8').strip())
        return {"header": message_header, "data": _client_socket.recv(message_length)}

    except Exception as e:
        print(f"Server error : {e}")
        return False


HEADER_LENGTH = 10
IP = socket.gethostname()
PORT = 12345

print("Please enter the party number...")
my_username = input("party number :")

print("Party is connecting to the server with IP: {0}, Port: {1}".format(IP, PORT))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))
client_socket.setblocking(True)

username = my_username.encode('utf-8')
username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
client_socket.send(username_header + username)

print("Party is initializing for learning...")
train_set, attribute_information, attributes_range, number_target_classes \
     = tools.get_chunk_of_data(my_username, 'Adult')

# initialization
# Settings
global_seed = 101
seed_common = 102
number_of_parties = 2  # 80
number_of_trees = 5
attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)
num_participating_parties = number_of_parties
Secure_Aggregation_SMC = True  # False # True if simulation of SMC part is required
Secure_Aggregation_Parameter_k = num_participating_parties-1  # can be changed to a value<num_participating_parties

party = generate_parties.generate(global_seed=global_seed, number_of_parties=number_of_parties,
                                  train_set=train_set,
                                  attribute_information=attribute_information,
                                  number_target_classes=number_target_classes,
                                  attributes_range=attributes_range,
                                  attribute_percentage=attribute_percentage,
                                  Secure_Aggregation_SMC=Secure_Aggregation_SMC,
                                  Secure_Aggregation_Parameter_k=Secure_Aggregation_Parameter_k,
                                  seed_common=seed_common,
                                  num_participating_parties=num_participating_parties,
                                  party_id=my_username)  # added username as party_id


print("Learning is started...")
while True:
    # Party waits for server request : {'check' or 'update_data_table'}
    message = client_socket.recv(1024)
    message = pickle.loads(message)

    if message['flag'] == "check":

        # print("client is processing check request")
        node_id = message['node_id']
        if node_id == 0:
            party.data_table = []
        branch = message['branch']
        true_temp, false_temp = party.check(node_id, branch)
        dic = {"true_temp": true_temp, "false_temp": false_temp}
        dict_message = pickle.dumps(dic)
        client_socket.send(dict_message)
        # print("client completed check request")

    if message['flag'] == "update_data_table":

        # print("client is processing update request")
        attribute_type = message['attribute_type']
        attribute_index = message['attribute_index']
        point_or_category = message['point_or_category']
        node_id = message['node_id']
        branch = message['branch']

        # create criterion object
        best_criterion = party.Criterion(attribute_type, attribute_index, point_or_category)

        party.update_data_table(best_criterion, node_id, branch)
        message = pickle.dumps("update is completed")
        client_socket.send(message)
        # print("client completed update request")
