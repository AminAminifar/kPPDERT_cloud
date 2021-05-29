import socket
from src import generate_parties
import tools
import pickle


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
train_set, test_set, attribute_information, attributes_range, number_target_classes \
     = tools.get_chunk_of_data(my_username, 'Adult')

party = generate_parties.generate(global_seed=13, train_set=train_set,
                                  attribute_information=attribute_information,
                                  number_target_classes=number_target_classes,
                                  attributes_range=attributes_range)

print("Learning is started...")
while True:
    # Party waits for server request : {'check' or 'update_data_table'}
    message = client_socket.recv(1024)
    message = pickle.loads(message)

    if message['flag'] == "check":

        # print("client is processing check request")
        node_id = message['node_id']
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
