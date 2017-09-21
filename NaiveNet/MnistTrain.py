from NaiveNet import NaiveNet
from matplotlib import pyplot
from time import sleep
import numpy as np
import pandas as pd

# read the mnist data. each row is an image. the first cell is the 
# ground truth, the next 28 * 28 cells are the pixels (in a 1D array)
mnist_test = pd.read_csv('.\mnist_test_10.csv', header=None)
mnist_train = pd.read_csv('.\mnist_train_100.csv', header=None)
num_test_images = mnist_test.shape[0]
num_train_images = mnist_train.shape[0]
input_nodes = mnist_test.shape[1] - 1
output_nodes = 10 # 0:9

# normalize the pixels to be (0, 1)
eps = 0.01 # don't want 1's or 0's in the neural net inputs
bit_depth = 8
max_value = (1 + eps) * (pow(2, bit_depth) - 1)
mnist_test.iloc[:, 1:] = (mnist_test.iloc[:, 1:] + eps) / max_value
mnist_train.iloc[:, 1:] = (mnist_train.iloc[:, 1:] + eps) / max_value

# intialize neural net
mnist_net = NaiveNet(input_nodes, 200, output_nodes, learn_rate=0.1)

# Train the neural net on the train images
epoch_iteration = 0
chatty = 0
img = 0
epochs = 20
sleeptime = 0
evaluations = []
print('\nTraining model...')
for epc in range(epochs):
    training_iteration = 0
    mnist_train = mnist_train.sample(frac=1) # randomly shuffle the training order

    for img in range(num_train_images):
        input_img = mnist_train.iloc[img, 1:] # e.g. 28 x 28 pixels as 1D array
        target_value = mnist_train.iloc[img, 0] # e.g. 7
        
        # a onehot encoding of the target_value
        target_classification = np.zeros(output_nodes) + input_img.min()
        target_classification[target_value] = input_img.max()
    
        # train
        input_layers = mnist_net.query(input_img)
        output_errors, outputs = mnist_net.train(target_classification, input_layers)
        
        training_iteration += 1
        if chatty > 0:
            print("Epoch: {}\nIteration: {}".format(epc, training_iteration))
            print("Input:{}\nEstimate: {}\nTarget:\n{}\nOutput:\n{}\nLoss:\n{}\n".format(
                    target_value, np.argmax(outputs), np.array(target_classification, ndmin=2).T, outputs, output_errors))
            sleep(sleeptime)
        
        if epc == epochs - 1:
            if np.argmax(outputs) == target_value: match = 1
            else: match = 0
            evaluations.append(match)

print("Final Train Accuracy: {} out of {} ({}%)".format(sum(evaluations), len(evaluations), 100 * (sum(evaluations) / len(evaluations))))

# Test
sleeptime = 3
evaluations = []
print('\nEvaluating model on test images...\n')
for img in range(num_test_images):
    input_img = mnist_test.iloc[img, 1:] # e.g. 28 x 28 pixels as 1D array
    target_value = mnist_test.iloc[img, 0] # e.g. 7
    
    # a onehot encoding of the target_value
    target_classification = np.zeros(output_nodes) + input_img.min()
    target_classification[target_value] = input_img.max()

    # test
    outputs = mnist_net.query(input_img)['fo']
    
    # output
    print("\nInput: {}\nOutput: {}".format(target_value, np.argmax(outputs)))
    imgplot = pyplot.imshow(input_img.values.reshape(28, 28), cmap='Greys')
    pyplot.show()
    if np.argmax(outputs) == target_value: match = 1
    else: match = 0
    evaluations.append(match)
    sleep(sleeptime)

print("Final Test Accuracy: {} out of {} ({}%)".format(sum(evaluations), len(evaluations), 100 * (sum(evaluations) / len(evaluations))))
