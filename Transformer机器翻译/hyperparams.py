# -*- coding: utf-8 -*-
class Hyperparams:

    # data
    source_train = 'corpora/train.cn'
    target_train = 'corpora/train.en'
    source_test = 'corpora/val.cn'
    target_test = 'corpora/val.en'
    
    # training
    batch_size = 32 
    lr = 0.0001 
    module = 'module' 
    
    # model
    maxlen = 30 
    min_cnt = 20 
    hidden_units = 512 
    num_blocks = 6 
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False 
    
    
    
    
