'''
Created on 8 Mar 2017

@author: Srikanth Ronanki
'''

import os
import regex as re
import tqdm
import time
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import seq2seq
from seq2seq.models import SimpleSeq2Seq

from io_funcs.binary_io import BinaryIOCollection

class kerasModels(object):
    
    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='softmax', dropout_rate=0.0, loss_function='categorical_crossentropy', optimizer='adam'):
        """ This function initialises a neural network
        
        :param n_in: Dimensionality of input features
        :param hidden_layer_size: The layer size for each hidden layer
        :param n_out: Dimensionality of output features
        :param hidden_layer_type: the activation types of each hidden layers, e.g., TANH, LSTM, GRU, BLSTM
        :param output_type: the activation type of the output layer, by default is 'LINEAR', linear regression.
        :param dropout_rate: probability of dropout, a float number between 0 and 1.
        :type n_in: Integer
        :type hidden_layer_size: A list of integers
        :type n_out: Integrer
        """
        
        self.n_in  = int(n_in)
        self.n_out = int(n_out)
        
        self.n_layers = len(hidden_layer_size)
        
        assert len(hidden_layer_size) == len(hidden_layer_type)
       
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_type = hidden_layer_type
        
        self.output_type   = output_type 
        self.dropout_rate  = dropout_rate
        self.loss_function = loss_function
        self.optimizer     = optimizer

        print "output_type   : "+self.output_type
        print "loss function : "+self.loss_function
        print "optimizer     : "+self.optimizer

    def define_baseline_model(self):
        # create model
        self.model = Sequential()

        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layer_size[i - 1]

            self.model.add(Dense(
                    output_dim=self.hidden_layer_size[i],
                    input_dim=input_size,
                    init='normal',
                    activation=self.hidden_layer_type[i]))
            #self.model.add(Dropout(self.dropout_rate))

        # add output layer
        self.final_layer = self.model.add(Dense(
            output_dim=self.n_out,
            input_dim=self.hidden_layer_size[-1],
            init='normal',
            activation=self.output_type.lower()))

        # Compile the model
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        print "model compiled successfully!!"

    def define_seq2seq_model(self):
        self.model = SimpleSeq2Seq(input_dim=self.n_in, hidden_dim=512, output_length=self.n_out, output_dim=self.n_out, depth=3)
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

def read_data_from_file_list(in_file_list, dim): 
    io_funcs = BinaryIOCollection()

    temp_set = np.empty((500000, dim))
     
    ### read file by file ###
    current_index = 0
    for i in tqdm.tqdm(range(len(in_file_list))):    
        in_file_name = in_file_list[i]
        in_features, frame_number = io_funcs.load_binary_file_frame(in_file_name, dim)
        temp_set[current_index:current_index+frame_number, ] = in_features
        current_index += frame_number
    
    temp_set = temp_set[0:current_index, ]
    
    return temp_set
            
def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

def read_file_list(file_name):
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    return  file_lists

            
if __name__ == "__main__":
   
    #### User configurable variables #### 
    
    merlin_dir = "/work/smg/v-srikanth/merlin"
    data_dir   = os.path.join(merlin_dir, "egs/fls_blizzard_2016/s1/experiments/blizzard_2016_full/acoustic_model/data")
    
    inp_dim = 100
    out_dim = 891

    inp_feat_dir = os.path.join(data_dir, 'nn_no_silence_lab_norm_'+str(inp_dim))
    out_feat_dir = os.path.join(data_dir, 'nn_norm_cls_'+str(out_dim))
    
    inp_file_ext = '.lab'
    out_file_ext = '.cmp'

    file_id_scp  = os.path.join(data_dir, 'file_id_list_full.scp')
    file_id_list = read_file_list(file_id_scp)

    train_file_number = 5500
    valid_file_number =  134
    test_file_number  =  253

    #### Train, valid and test file lists #### 

    train_id_list = file_id_list[0:train_file_number]
    valid_id_list = file_id_list[train_file_number:train_file_number+valid_file_number]
    test_id_list  = file_id_list[train_file_number+valid_file_number:train_file_number+valid_file_number+test_file_number]
    
    inp_train_file_list = prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
    out_train_file_list = prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)
    
    inp_test_file_list = prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
    out_test_file_list = prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)
    
    print 'preparing train_x from input feature files...'
    train_x = read_data_from_file_list(inp_train_file_list, inp_dim)
     
    print 'preparing train_y from output feature files...'
    train_y = read_data_from_file_list(out_train_file_list, out_dim)

    print 'preparing test_x from input feature files...'
    test_x = read_data_from_file_list(inp_test_file_list, inp_dim)
     
    print 'preparing test_y from output feature files...'
    test_y = read_data_from_file_list(out_test_file_list, out_dim)
    
    #### define Model, train and test ####
    
    hidden_layer_type = ['tanh','tanh','tanh','tanh','tanh','tanh']
    hidden_layer_size = [ 1024 , 1024 , 1024 , 1024 , 1024 , 1024 ]

    output_type   ='softmax'
    dropout_rate  =0.0
    loss_function ='categorical_crossentropy'
    optimizer     ='adam'

    diph_classifier = kerasModels(inp_dim, hidden_layer_size, out_dim, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

    diph_classifier.define_baseline_model()
    #diph_classifier.define_seq2seq_model()
    model = diph_classifier.model

    #### train the model ####
    
    model.fit(train_x, train_y, batch_size=256, nb_epoch=7)

    #### evaluate the model ####
    
    print "evaluating the model on held-out test data..."
    scores = model.evaluate(test_x, test_y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    #### calculate predictions ####
    
    #predictions = model.predict(test_x)
    #print predictions.shape
