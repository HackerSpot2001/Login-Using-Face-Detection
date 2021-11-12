#!/usr/bin/python3
import os
from loginSystem import login
from registrationSystem import trainModel

if len(os.listdir("collections")) != 0 :
    login()

else:
    trainModel()
