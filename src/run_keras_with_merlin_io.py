################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

import os
import sys
import time

from keras_lib import configuration
from keras_lib import data_utils
from keras_lib.train import TrainKerasModels

def main(cfg):
  
    ###################################################
    ########## User configurable variables ############ 
    ###################################################

    work_dir = cfg.work_dir 
    data_dir = cfg.data_dir
    
    inp_feat_dir = cfg.inp_feat_dir
    out_feat_dir = cfg.out_feat_dir
    
    model_dir    = cfg.model_dir
    stats_dir    = cfg.stats_dir
    gen_dir      = cfg.gen_dir

    ### Input-Output ###

    inp_dim = cfg.inp_dim
    out_dim = cfg.out_dim

    inp_file_ext = cfg.inp_file_ext
    out_file_ext = cfg.out_file_ext

    inp_norm = cfg.inp_norm
    out_norm = cfg.out_norm

    ### define train, valid, test ###

    train_file_number = cfg.train_file_number
    valid_file_number = cfg.valid_file_number
    test_file_number  = cfg.test_file_number
    
    #### Train, valid and test file lists #### 
    
    file_id_list = data_utils.read_file_list(cfg.file_id_scp)

    train_id_list = file_id_list[0:train_file_number]
    valid_id_list = file_id_list[train_file_number:train_file_number+valid_file_number]
    test_id_list  = file_id_list[train_file_number+valid_file_number:train_file_number+valid_file_number+test_file_number]
    
    inp_train_file_list = data_utils.prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
    out_train_file_list = data_utils.prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)
    
    inp_test_file_list = data_utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
    out_test_file_list = data_utils.prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)
  
    gen_test_file_list = data_utils.prepare_file_path_list(test_id_list, cfg.pred_feat_dir, out_file_ext)
 
    #### define model params ####

    inp_scaler = None
    out_scaler = None
        
    hidden_layer_type = cfg.hidden_layer_type
    hidden_layer_size = cfg.hidden_layer_size
       
    batch_size    = cfg.batch_size
    training_algo = cfg.training_algo

    output_layer_type = cfg.output_layer_type
    loss_function     = cfg.loss_function
    optimizer         = cfg.optimizer
    
    num_of_epochs = cfg.num_of_epochs
    dropout_rate  = cfg.dropout_rate

    json_model_file = cfg.json_model_file
    h5_model_file   = cfg.h5_model_file
         
    ###################################################
    ########## End of user-defined variables ##########
    ###################################################

    #### Define keras models class ####
    keras_models = TrainKerasModels(inp_dim, hidden_layer_size, out_dim, hidden_layer_type, output_layer_type, dropout_rate, loss_function, optimizer)
    
    if cfg.NORMDATA:
        ### normalize train data ###
        if os.path.isfile(cfg.inp_stats_file) and os.path.isfile(cfg.out_stats_file):    
            inp_scaler = data_utils.load_norm_stats(cfg.inp_stats_file, inp_dim, method=inp_norm)
            out_scaler = data_utils.load_norm_stats(cfg.out_stats_file, out_dim, method=out_norm)
        else:
            print('preparing train_x, train_y from input and output feature files...')
            train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list, out_train_file_list, 
                                                                            inp_dim, out_dim, sequential_training=cfg.sequential_training)
            
            print('computing norm stats for train_x...')
            inp_scaler = data_utils.compute_norm_stats(train_x, cfg.inp_stats_file, method=inp_norm)
            
            print('computing norm stats for train_y...')
            out_scaler = data_utils.compute_norm_stats(train_y, cfg.out_stats_file, method=out_norm)

        
    if cfg.TRAINMODEL:
        #### define the model ####
        if not cfg.sequential_training:
            keras_models.define_feedforward_model()
        elif cfg.stateful:
            keras_models.define_stateful_model(batch_size=batch_size)
        else:
            keras_models.define_sequence_model()
        
        #### load the data ####
        print('preparing train_x, train_y from input and output feature files...')
        train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list, out_train_file_list, 
                                                                            inp_dim, out_dim, sequential_training=cfg.sequential_training)

        #### norm the data ####
        data_utils.norm_data(train_x, inp_scaler, sequential_training=cfg.sequential_training)
        data_utils.norm_data(train_y, out_scaler, sequential_training=cfg.sequential_training)
        
        #### train the model ####
        print('training...')
        if not cfg.sequential_training:
            ### Train feedforward model ###
            keras_models.train_feedforward_model(train_x, train_y, batch_size=batch_size, num_of_epochs=num_of_epochs, shuffle_data=cfg.shuffle_data) 
        else:
            ### Train recurrent model ###
            keras_models.train_sequence_model(train_x, train_y, train_flen, batch_size=batch_size, num_of_epochs=num_of_epochs, 
                                                                                        shuffle_data=cfg.shuffle_data, training_algo=training_algo) 

        #### store the model ####
        keras_models.save_model(json_model_file, h5_model_file)
   
    if cfg.TESTMODEL: 
        #### load the model ####
        keras_models.load_model(json_model_file, h5_model_file)

        #### load the data ####
        print('preparing test_x from input feature files...')
        test_x, test_flen = data_utils.read_test_data_from_file_list(inp_test_file_list, inp_dim)
     
        #### norm the data ####
        data_utils.norm_data(test_x, inp_scaler)
        
        #### compute predictions ####
        keras_models.predict(test_x, out_scaler, gen_test_file_list, cfg.sequential_training)


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        logger.critical('usage: python run_keras_with_merlin_io.py [config file name]')
        sys.exit(1)

    # create a configuration instance
    # and get a short name for this instance
    cfg = configuration.configuration()
    
    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    
    print("--- Job started ---")
    start_time = time.time()
    
    # main function
    main(cfg)

    (m, s) = divmod(int(time.time() - start_time), 60) 
    print("--- Job completion time: %d min. %d sec ---" % (m, s)) 

    sys.exit(0)

