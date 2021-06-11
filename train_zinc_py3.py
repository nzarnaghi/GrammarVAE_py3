from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from models.model_zinc_py3 import MoleculeVAE
import tensorflow as tf

import h5py
import zinc_grammar as G

MAX_LEN = 277
'''
LATENT = 25
EPOCHS = 50
BATCH = 600
'''
rules = G.gram.split('\n')
DIM = len(rules)
LATENT = 2 #56 # 2
EPOCHS = 100
BATCH = 500

NCHARS = len(G.GCFG.productions())



print("rules")
print(DIM)

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

def main():

    # 0. load dataset
    h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    
    
    
    # 1. split into train/test, we use test set to check reconstruction error and the % of
    # samples from prior p(z) that are valid
    
    print("shape")
    print(np.shape(data))
    XTE = data[0:5000]
    #XTR = data[500:]
    
    print("type")
    print(type(data))
    print(data[1])
    #data = int(data)
    data2 = data.astype(int)
    print("data2")
    print(data2[1])
    
    print("rules")
    print(DIM)

    # 1. get any arguments and define save file, then create the VAE model
    args = get_arguments()
    params = {'hidden': 501, 'dense': 435, 'conv1': 9, 'conv2': 9, 'conv3': 10}
    model_save = 'zinc_vae_grammar_h' + str(params['hidden']) + '_c234_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_batchB.hdf5'
    model = MoleculeVAE()

    # 2. if this results file exists already load it
    if os.path.isfile(model_save):
        model.load(rules, model_save, latent_rep_size = args.latent_dim, hypers = params)
    else:
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim, hypers = params)

    # 3. only save best model found on a 10% validation set
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    # 4. fit the vae
    model.autoencoder.fit(
        XTE,
        XTE,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = BATCH,
        callbacks = [checkpointer, reduce_lr],
        validation_split = 0.1
    )

if __name__ == '__main__':
    main()
