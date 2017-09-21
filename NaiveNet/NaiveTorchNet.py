import numpy as np
from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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
    binary_net = NaiveTorchNet(3, 50, 8, learn_rate=10)

    # train
    epochs = 50
    eps = 0.01 # don't want 1's or 0's in the neural net inputs
    max_value = 1 + 2*eps
    sleep_time = 1 / epochs
    evaluations = []
    chatty = 0
    for epc in range(epochs):
        evaluations.clear()

        # create minibatch of all 8 encodings per epoch
        inputs = torch.cat([Variable(torch.Tensor((np.array(val[0]) + eps) / max_value).unsqueeze(1).transpose(-1, 0)) for val in binary_encoding], 0)
        target_classification = torch.cat([Variable(torch.Tensor((np.array(val[1]) + eps) / max_value).unsqueeze(1).transpose(-1, 0)) for val in binary_encoding], 0)
        
        target_value = [np.argmax(target) for target in target_classification.data.tolist()]

        input_layers = binary_net.query(inputs)
        output_errors, outputs = binary_net.learn(target_classification, input_layers)

        for ii in range(len(binary_encoding)):
            print("Epoch: {}, Input: {}, Estimate: {}, Loss: {}".format(epc, target_value[ii], outputs.data.max(0)[1][ii], output_errors.data[0]))
            if chatty > 0: print("Target:\n{}\nOutput:\n{}\n".format(target_classification[ii,:], outputs[ii,:]))
            sleep(sleep_time)

            if outputs.data.max(0)[1][ii] == target_value[ii]: match = 1
            else: match = 0
            evaluations.append(match)

        print("=====\nEPOCH ACCURACY: {} out of {} ({}%)\n=====\n".format(sum(evaluations), len(evaluations), 100 * (sum(evaluations) / len(evaluations))))
        sleep(3 * sleep_time)

    # when done, print everything
    for ii in range(len(binary_encoding)):
        print("Epoch: {}, Input: {}, Estimate: {}, Loss: {}".format(epc, target_value[ii], outputs.data.max(0)[1][ii], output_errors.data[0]))
        print("Target:\n{}\nOutput:\n{}\n".format(target_classification[ii,:], outputs[ii,:]))
        print("=====\nEPOCH ACCURACY: {} out of {} ({}%)\n=====\n".format(sum(evaluations), len(evaluations), 100 * (sum(evaluations) / len(evaluations))))

class NaiveTorchNet(nn.Module):
    """A reimplementation of from-scratch NaiveNet using PyTorch"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate=0.1):
        super().__init__()
        self.hidden = nn.Linear(input_nodes, hidden_nodes, bias=False)
        self.output = nn.Linear(hidden_nodes, output_nodes, bias=False)

        self.lr = learn_rate
        self.activation_function = nn.Sigmoid()
        self.optimizer = optim.SGD(self.parameters(), lr=learn_rate)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        """Overrides the built in"""
        x = self.activation_function(self.hidden(x))
        x = self.activation_function(self.output(x))
        return x

    def query(self, inputs):
        """Takes an input to the net and returns an output via forward computation"""
        if type(inputs) != torch.autograd.variable.Variable: inputs = Variable(torch.Tensor(inputs))
        return { 'i': inputs, 'fo': self.forward(inputs) }

    def learn(self, targets, input_layers):
        if type(targets) != torch.autograd.variable.Variable: targets = Variable(torch.Tensor(targets))
        final_outputs = input_layers['fo']
        output_errors = self.loss_function(final_outputs, targets)

        self.optimizer.zero_grad()
        output_errors.backward()
        self.optimizer.step()

        return output_errors, final_outputs

if __name__ == '__main__':
    main()