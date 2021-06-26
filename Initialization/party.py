from __future__ import division
import pandas as pd
import numpy as np
import csv
from Crypto.PublicKey import RSA
from Crypto import Random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import math
import random

class Party:
    def __init__(self, num_parties):
        # generate public and private key (RSA encryption)
        self.key = RSA.generate(2048)
        self.publickey = self.key.publickey()
        self.n = num_parties
        self.k = num_parties
        self.seeds = None
        self.encrypted_seeds = None
        self.decrypted_seeds = None

    def share_public_key(self):
        """This function shares public key to mediator"""
        return self.publickey

    def generate_seeds(self):
        temp = []
        for i in range(self.n - 1):
            temp.append(random.randint(1, 10 ** 5))
        self.seeds = np.array(temp)

    def encrypt_seeds(self, pub_keys):
        temp = []
        for i in range(self.n - 1):
            pub_key = pub_keys[i]
            seed = self.seeds[i]
            temp.append(pub_key.encrypt(long(seed), '')[0])
        self.encrypted_seeds = np.array(temp)

    def decrypt_seeds(self, received_seeds):
        temp = []
        for i in range(self.k - 1):
            seed = received_seeds[i]
            temp.append(self.key.decrypt(seed))
        self.decrypted_seeds = np.array(temp)


