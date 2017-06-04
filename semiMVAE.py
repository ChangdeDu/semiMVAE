from semiClass import GenerativeClassifier
import numpy as np
from keras.utils import np_utils
from scipy.io import loadmat
    
#############################
''' Experiment Parameters '''
#############################     
num_batches = 10        #Number of minibatches in a single epoch
dim_z = 30              #Dimensionality of latent variable (z)
epochs = 1001           #Number of epochs through the full dataset
learning_rate = 3e-4    #Learning rate of ADAM
beta = 5                #Discriminatory factor 
l2_loss = 1e-6          #L2 Regularisation weight
zeta_reg = 1e8
seed = 31415            #Seed for RNG

#Neural Networks parameterising p(x1|z,y), p(x2|z,y), q(z|x,y) and q(y|x)
hidden_layers_px1 = [ 50 ]
hidden_layers_px2 = [ 50 ]
hidden_layers_qz = [ 50 ]
hidden_layers_qy = [ 50 ]

####################
''' Load Dataset '''
####################
data=loadmat('/home/cddu/Desktop//deap_data_x1.mat')
enc_x1_mean = data['enc_mean'].astype('float32')
enc_x1_var  = data['enc_var'].astype('float32')
Y = data['Y']

data=loadmat('/home/cddu/Desktop/deap_data_x2.mat')
enc_x2_mean = data['enc_mean'].astype('float32')
enc_x2_var  = data['enc_var'].astype('float32')
Y = data['Y']


filtered=loadmat('filter_idx_3_0.mat')
idx = filtered['filter_idx']
num = idx.shape[0]
idx = idx.reshape(num) - 1
enc_x1_mean = enc_x1_mean[idx,:]
enc_x1_var = enc_x1_var[idx,:]
enc_x2_mean = enc_x2_mean[idx,:]
enc_x2_var = enc_x2_var[idx,:]
Y = Y[idx,:]

# convert class vectors to binary class matrices
nb_classes = len(np.unique(Y))
Y = np_utils.to_categorical(Y, nb_classes)

loop=20
acc_test = np.zeros((loop,1))
acc_train = np.zeros((loop,1))
for iter in range(loop):
    # shuffle data
    randomize = np.arange(Y.shape[0])
    np.random.shuffle(randomize)
    enc_x1_mean = enc_x1_mean[randomize,:]
    enc_x1_var = enc_x1_var[randomize,:]
    enc_x2_mean = enc_x2_mean[randomize,:]
    enc_x2_var = enc_x2_var[randomize,:]

    Y = Y[randomize,:]
    
    num_train = np.round(0.8 * enc_x1_mean.shape[0]).astype('int')
    num_test = np.round(0.1 * enc_x1_mean.shape[0]).astype('int')
    num_val = np.round(0.1 * enc_x1_mean.shape[0]).astype('int')
    
    rate = 0.01
    num_lab = np.round(rate * num_train).astype('int')
    lab_idx = range(0, num_lab) 
    
    rate_u = 0.99
    num_u = np.round((1-rate_u) * num_train).astype('int')
    ulab_idx = range(num_u, num_train)
    valid_idx = range(num_train,num_train+num_val)
    test_idx = range(num_train+num_val,num_train+num_val+num_test)
    
    
    enc_x1_lab_mean = enc_x1_mean[lab_idx,:]
    enc_x1_lab_var = enc_x1_var[lab_idx,:]
    enc_x1_ulab_mean = enc_x1_mean[ulab_idx,:]
    enc_x1_ulab_var = enc_x1_var[ulab_idx,:]
    
    enc_x2_lab_mean = enc_x2_mean[lab_idx,:]
    enc_x2_lab_var = enc_x2_var[lab_idx,:]
    enc_x2_ulab_mean = enc_x2_mean[ulab_idx,:]
    enc_x2_ulab_var = enc_x2_var[ulab_idx,:]
    
    enc_x1_test_mean = enc_x1_mean[test_idx,:]
    enc_x1_test_var = enc_x1_var[test_idx,:]
    enc_x1_valid_mean = enc_x1_mean[valid_idx,:]
    enc_x1_valid_var = enc_x1_var[valid_idx,:]
    
    enc_x2_test_mean = enc_x2_mean[test_idx,:]
    enc_x2_test_var = enc_x2_var[test_idx,:]
    enc_x2_valid_mean = enc_x2_mean[valid_idx,:]
    enc_x2_valid_var = enc_x2_var[valid_idx,:]
    
    y_lab = Y[lab_idx,:]
    y_test = Y[test_idx,:]
    y_valid = Y[valid_idx,:]
    
    data_lab_x1 = np.hstack( [ enc_x1_lab_mean, enc_x1_lab_var ] )
    data_ulab_x1 = np.hstack( [ enc_x1_ulab_mean, enc_x1_ulab_var ] )
    data_valid_x1 = np.hstack( [enc_x1_valid_mean, enc_x1_valid_var] )
    data_test_x1 = np.hstack( [enc_x1_test_mean, enc_x1_test_var] )
    
    data_lab_x2 = np.hstack( [ enc_x2_lab_mean, enc_x2_lab_var ] )
    data_ulab_x2 = np.hstack( [ enc_x2_ulab_mean, enc_x2_ulab_var ] )
    data_valid_x2 = np.hstack( [enc_x2_valid_mean, enc_x2_valid_var] )
    data_test_x2 = np.hstack( [enc_x2_test_mean, enc_x2_test_var] )
    
    ################
    ''' Load MVAE '''
    ################
    
    dim_x1 = data_lab_x1.shape[1] / 2
    dim_x2 = data_lab_x2.shape[1] / 2
    dim_y = y_lab.shape[1]
    num_examples = data_lab_x1.shape[0] + data_ulab_x1.shape[0]

    ###################################
    ''' Train Generative Classifier '''
    ###################################
    
    GC = GenerativeClassifier(  dim_x1, dim_x2, dim_z, dim_y,
                                num_examples, num_lab, num_batches,
                                hidden_layers_px1    = hidden_layers_px1,
                                hidden_layers_px2    = hidden_layers_px2,
                                hidden_layers_qz    = hidden_layers_qz, 
                                hidden_layers_qy    = hidden_layers_qy,
                                beta               = beta,
                                l2_loss              = l2_loss,
                                zeta_reg             = zeta_reg)
    
    GC.train(   x1_labelled     = data_lab_x1, 
                x2_labelled     = data_lab_x2, 
                y               = y_lab, 
                x1_unlabelled   = data_ulab_x1,
                x2_unlabelled   = data_ulab_x2,
                x1_valid        = data_valid_x1,
                x2_valid        = data_valid_x2,
                y_valid         = y_valid,
                epochs          = epochs, 
                learning_rate   = learning_rate,
                seed            = seed,
                stop_iter       = 30,
                print_every     = 10,
                load_path       = None )
    
    
    ############################
    ''' Evaluate on Test Set '''
    ############################
    
    GC_eval = GenerativeClassifier( dim_x1, dim_x2, dim_z, dim_y,
                                num_examples, num_lab, num_batches,
                                hidden_layers_px1    = hidden_layers_px1,
                                hidden_layers_px2    = hidden_layers_px2,
                                hidden_layers_qz     = hidden_layers_qz, 
                                hidden_layers_qy     = hidden_layers_qy,
                                beta                = beta,
                                l2_loss              = l2_loss,
                                zeta_reg             = zeta_reg)
    
    with GC_eval.session:
        GC_eval.saver.restore( GC_eval.session, GC.save_path )
        Y_pred = GC_eval.predict_labels( data_test_x1, data_test_x2, y_test )
        

   
