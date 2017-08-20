# with reference to: https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G
import numpy as np
from numpy import random as rand
import scipy.special
from time import sleep

def main():
    print('\nTesting 3 bit binary encoding classification')
    
    # (out: [onehot], in: [binary])
    binary_encoding = [
        ([0,0,0], [1,0,0,0,0,0,0,0]),
        ([0,0,1], [0,1,0,0,0,0,0,0]),
        ([0,1,0], [0,0,1,0,0,0,0,0]),
        ([0,1,1], [0,0,0,1,0,0,0,0]),
        ([1,0,0], [0,0,0,0,1,0,0,0]),
        ([1,0,1], [0,0,0,0,0,1,0,0]),
        ([1,1,0], [0,0,0,0,0,0,1,0]),
        ([1,1,1], [0,0,0,0,0,0,0,1])]

    # 8 integer input, 4 hidden layer nodes, 3 bit classification output
    binary_net = NaiveNet(3, 50, 8)

    # train
    epochs = 50
    eps = 0.01 # don't want 1's or 0's in the neural net inputs
    max_value = 1 + 2*eps
    sleep_time = 10 / epochs
    evaluations = []
    for epc in range(epochs):
        training_iteration = 0

        for val in binary_encoding:
            inputs = (np.array(val[0]) + eps) / max_value
            target_classification = (np.array(val[1]) + eps) / max_value
            target_value = np.argmax(target_classification)

            # process then optimize
            input_layers = binary_net.query(inputs)
            output_errors, outputs = binary_net.train(target_classification, input_layers)
    
            training_iteration += 1
            print("Epoch: {}\nIteration: {}".format(epc, training_iteration))
            print("Input:{}\nEstimate: {}\nTarget:\n{}\nOutput:\n{}\nLoss:\n{}\n".format(
                    target_value, np.argmax(outputs), np.array(target_classification, ndmin=2).T, outputs, output_errors))
            sleep(sleep_time)

            if epc == epochs - 1:
                if np.argmax(outputs) == target_value: match = 1
                else: match = 0
                evaluations.append(match)

    print("Final Train Accuracy: {} out of {} ({}%)".format(sum(evaluations), len(evaluations), 100 * (sum(evaluations) / len(evaluations))))

class NaiveNet():
    """A naive 3 layer neural network without biases."""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate=0.3, initial_weights='gaussian', loss_function='difference'):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.activation_function = lambda x: scipy.special.expit(x) # sigmoid / logistic
        self.lr = learn_rate

        self.__set_weights(initial_weights)
        self.__set_loss_function(loss_function)
        
    def __repr__(self):
        members = [(k, str(v)) for k, v in vars(self).items() if not k.startswith('_')]
        printable = ['    {}: {}'.format(m[0], m[1]) for m in members]
        return '{}{}{}'.format('NaiveNet(\n', '\n'.join(printable), '\n)')

    def __set_weights(self, initial_weights):
        if initial_weights == 'uniform':
            self.wih = rand.rand(self.inodes, self.hnodes) - 0.5
            self.who = rand.rand(self.hnodes, self.onodes) - 0.5
        elif initial_weights == 'gaussian':
            self.wih = rand.normal(0.0, pow(self.inodes, -0.5), [self.hnodes, self.inodes])
            self.who = rand.normal(0.0, pow(self.hnodes, -0.5), [self.onodes, self.hnodes])
        else:
            raise RuntimeError('initial_weights: "{}" not supported.'.format(initial_weights))

    def __set_loss_function(self, loss_function):
        if loss_function == 'difference':
            self.loss_function = lambda x, y: x - y
        elif loss_function == 'euclidean':
            self.loss_function = lambda x, y: abs(x - y)
            raise RuntimeError('does not work - need to fix')
        elif loss_function == 'squared':
            self.loss_function = lambda x, y: pow(x - y, 2)
            raise RuntimeError('does not work - need to fix')
        else:
            raise RuntimeError('loss_function_type: "{}" not supported.'.format(loss_function))

    def query(self, inputs):
        """Takes an input to the net and returns an output via forward computation"""

        # convert inputs list to 2d array
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return { 'i': inputs, 'hi': hidden_inputs, 'ho': hidden_outputs, 'fi': final_inputs, 'fo': final_outputs }

    def train(self, targets, input_layers):
        inputs = input_layers['i']
        targets = np.array(targets, ndmin=2).T # convert targets list to 2d array
        hidden_outputs = input_layers['ho']
        final_outputs = input_layers['fo']

        # apply the loss function to the output to get the final errors
        output_errors = self.loss_function(targets, final_outputs)

        # backpropogate the errors - split by weights per node then recombine
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the output layer weights via gradient descent
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)

        # update the hidden layer weights via gradient descent
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

        return output_errors, final_outputs

if __name__ == '__main__':
    main()
    