import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def alexnet(width, height, lr):
    
    network = input_data(shape=[None, width, height, 1], name='input')

    network = conv_2d(network, 32,3, activation='relu', bias=True)
    network = max_pool_2d(network,2)
    network = local_response_normalization(network)


    network = conv_2d(network, 64, 3, activation='relu', bias=True)
    network = max_pool_2d(network,2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu', bias=True)
    
    #network = max_pool_2d(network, [1,2,2,1], strides=[1,2,2,1])
    #network = local_response_normalization(network)
    
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1, activation='sigmoid')


    network = regression(network, optimizer='adam', 
            loss='mean_square',
            learning_rate=lr, name='targets')

    
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model
