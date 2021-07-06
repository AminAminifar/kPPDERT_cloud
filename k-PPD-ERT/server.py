import socket
import src.server_class
import src.Server_Parties_Interface
from tools import get_information_of_data
import src.Prediction_and_Classification_Performance
import numpy as np

HEADER_LENGTH = 10
IP = socket.gethostname()
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((IP, PORT))
server_socket.listen()

# socket_list = [server_socket]
socket_list = []
number_of_parties = 2
parties = {}


print("Server initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))


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


print("Server is waiting for parties, number of parties are: {0}".format(number_of_parties))
# Register all parties
i = 1
while i <= number_of_parties:
    client_socket, client_address = server_socket.accept()
    party = receive_message(client_socket)
    socket_list.append(client_socket)
    parties[client_socket] = party
    i = i + 1

    print(f"Accepted new connection from {client_address[0]} : {client_address[1]} "
          f"username: {party['data'].decode('utf-8')}")


print("All parties are accepted!")

interface = src.Server_Parties_Interface.Interface(parties, socket_list)

test_set, attribute_information, attributes_range, number_target_classes = get_information_of_data('Adult')

# initialization
# Settings
global_seed = 101
attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)
Secure_Aggregation_SMC = True  # False # True if simulation of SMC part is required
included_parties_indices = np.array(range(0,number_of_parties))

server = server_class.server(global_seed=global_seed, attribute_range=attributes_range,
                             attribute_info=attribute_information,
                             num_target_classes=number_target_classes,
                             aggregator_func=Interface.aggregator,
                             parties_update_func=Interface.parties_update,
                             attribute_percentage=attribute_percentage,
                             included_parties_indices=included_parties_indices,
                             Secure_Aggregation_SMC=Secure_Aggregation_SMC)  # parties_reset_func=Interface.parties_reset

print("learning is started...")
learned_model = server.make_tree_group()

print("Classification performance for several trees (F1 score/F1 score macro):",
      src.Prediction_and_Classification_Performance.ensemble_f1_score_for_a_set(learned_model, test_set))
