import socket
import src.server_class
import src.Server_Parties_Interface
from tools import get_information_of_data
import src.Prediction_and_Classification_Performance
import numpy as np
import sys
import pickle
import select

HEADER_LENGTH = 10
IP = socket.gethostname()
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((IP, PORT))
server_socket.listen()

socket_list = []
parties = {}

def initial_scenario(scenario):
    scenarios = {
        '1': 2, '2':5, '3':10, '4':20
    }
    return scenarios.get(str(scenario))

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

def main():

    scenario = int(sys.argv[1:][0])
    number_of_parties = initial_scenario(scenario)
    datasets = ['breast', 'Depression', 'heart', 'psykose']

    print("Server initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))

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

    for dataset in datasets:

        print('Dataset : {}'.format(dataset))

        test_set, attribute_information, attributes_range, number_target_classes = get_information_of_data(scenario, dataset)

        # initialization
        # Settings
        global_seed = 101
        attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)
        Secure_Aggregation_SMC = True  # False # True if simulation of SMC part is required
        included_parties_indices = np.array(range(0,number_of_parties))

        server = src.server_class.server(global_seed=global_seed, attribute_range=attributes_range,
                                    attribute_info=attribute_information,
                                    num_target_classes=number_target_classes,
                                    aggregator_func=interface.aggregator,
                                    parties_update_func=interface.parties_update,
                                    attribute_percentage=attribute_percentage,
                                    included_parties_indices=included_parties_indices,
                                    Secure_Aggregation_SMC=Secure_Aggregation_SMC)  # parties_reset_func=Interface.parties_resetz

        print("learning is started...")
        learned_model = server.make_tree_group()

        print("Classification performance:")
        src.Prediction_and_Classification_Performance.print_results(learned_model, test_set)

        for party in parties:
            message = {"flag": "leave"}
            party.send(pickle.dumps(message))

        flag = True
        j=0
        while flag:
            read_sockets, _, exception_sockets = select.select(socket_list, [], socket_list)

            for notified_socket in read_sockets:
                if j != len(parties):
                    message = notified_socket.recv(2048)
                    message = pickle.loads(message)
                    status = message["status"]
                    if status == 'leave':
                        j = j + 1
                if j == len(parties):
                    flag = False

if __name__ == '__main__':
    main()