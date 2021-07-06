
import pandas as pd
import numpy as np
import csv
from Cryptodome.PublicKey import RSA
from Cryptodome import Random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import math
import random
from Cryptodome.Cipher import PKCS1_OAEP

# key = RSA.generate(2048)
# publickey = key.publickey()

# seed = 100
#
# en_seed = publickey.encrypt(long(seed), '')[0]
#
# print("en_seed: ", en_seed)
# de_seed = key.decrypt(en_seed)
# print("de_seed: ", de_seed)
#
#
# seed2 = 5
# en_seed2 = publickey.encrypt(long(seed2), '')[0]
# mul = en_seed * en_seed2
# print("mul: ", mul)
# de_mul = key.decrypt(mul)
# print("de_mul: ", de_mul)

# cipher = PKCS1_OAEP.new(key)
# text = b'The secret I want to send.'
# en_text = cipher.encrypt(text)
# print("en_text: ", en_text)
# de_text = cipher.encrypt(en_text)
# print("de_text: ", de_text)

keyPair = RSA.generate(3072)
pubKey = keyPair.publickey()
# pubKeyPEM = pubKey.exportKey()
# print(pubKeyPEM.decode('ascii'))
# privKeyPEM = keyPair.exportKey()
# print(privKeyPEM.decode('ascii'))


msg = b'100'
encryptor = PKCS1_OAEP.new(pubKey)
print(pubKey)
encrypted = encryptor.encrypt(msg)
print("Encrypted:",encrypted)


decryptor = PKCS1_OAEP.new(keyPair)
decrypted = decryptor.decrypt(encrypted)
print('Decrypted:',decrypted)
print('num:',int(decrypted))

# en_text = pubKey.encrypt('hhh')
# print("en_text: ", en_text)
# de_text = privKeyPEM.encrypt(en_text)
# print("de_text: ", de_text)


# f = open('secret.txt','wb')
# f.write(key.export_key('PEM'))
# f.close()
# ...
# >>> f = open('mykey.pem','r')
# >>> key = RSA.import_key(f.read())