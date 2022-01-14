import numpy
import math
import random
from tqdm import tqdm
import argparse

def sigmoid(vector):
    """
    sigmoid = numpy.where(vector < 0, numpy.exp(vector)/(1 + numpy.exp(vector)), 1/(1 + numpy.exp(-vector)))
    return sigmoid
    """
    return 1.0 / (1.0 + numpy.exp(-vector))

def d_sigmoid(vector):
    return sigmoid(vector) * (1 - sigmoid(vector))#(numpy.exp(-vector))/((numpy.exp(-vector)+1)**2)
"""
def softmax(vector):
    exponents = numpy.exp(vector - vector.max())
    return exponents / numpy.sum(exponents)

def d_softmax(vector):
    exponents = numpy.exp(vector - vector.max())
    return exponents / numpy.sum(exponents) * (1 - exponents / numpy.sum(exponents))
"""

def softmax(vector):
    exponents = numpy.exp(vector - vector.max())
    return exponents / numpy.sum(exponents)

def d_softmax(vector):
    exponents = numpy.exp(vector - vector.max())
    return exponents/ numpy.sum(exponents) * (1 - exponents / numpy.sum(exponents))

def compute_batch_loss(pred, ground_truth):
    return -numpy.sum(numpy.dot(ground_truth.T, numpy.log(pred+0.000001)))


SIGMOID = numpy.vectorize(sigmoid)
D_SIGMOID = numpy.vectorize(d_sigmoid)
D_SOFTMAX = numpy.vectorize(d_softmax)


class Net:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = []
        self.outputs = []
        self.inputs = []
        self.deltas = []
        self.gradients = []
        for i in range(0, len(sizes)-2):
            self.weights.append(numpy.random.randn(sizes[i+1]+1, sizes[i]+1) * numpy.sqrt(1. / sizes[i+1]+1))
            self.gradients.append(numpy.zeros((sizes[i+1]+1, sizes[i]+1)))
            #print("Weights[{}] = {}".format(i, self.weights[i]))
            #print("Gradients[{}] = {}".format(i, self.gradients[i].shape))
        self.weights.append(numpy.random.rand(sizes[-1], sizes[-2]+1) * numpy.sqrt(1. / sizes[-1]+1))
        self.gradients.append(numpy.zeros((sizes[-1], sizes[-2]+1)))
        #print("Weights[{}] = {}".format(len(sizes)-2, self.weights[-1]))
        #print("Gradients[{}] = {}".format(len(sizes)-1, self.gradients[-1].shape))

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def reset_gradients(self):
        self.gradients = []
        for i in range(0, len(self.sizes)-2):
            self.gradients.append(numpy.zeros((self.sizes[i+1]+1, self.sizes[i]+1)))
            #print("Gradients[{}] = {}".format(i, self.gradients[i].shape))
        self.gradients.append(numpy.zeros((self.sizes[-1], self.sizes[-2]+1)))

    def forward(self, sample):
        #print("=========================")
        s = sample
        self.outputs.append(s)
        for i, matrix in enumerate(self.weights):
            #print("W[{}] = {}".format(i, matrix))
            #print("s[{}] = {}".format(i, s))
            z = numpy.dot(s, matrix.T)
            #inputs = inputs / inputs.max()
            #inputs = inputs.reshape((inputs.shape[0], 1))
            #print("z[{}] = {}".format(i, z.shape))
            #input()
            self.inputs.append(z)
            if i == len(self.weights)-1:
                s = numpy.apply_along_axis(softmax, 1, z)
                #print("···············\n{}\n···············".format(s))
            else:
                s = numpy.apply_along_axis(sigmoid, 1, z)
            #prev = prev.reshape((prev.shape[0], 1))
            self.outputs.append(s)
            #print("s[{}] = {}".format(i, s.shape))
        return s

    
    def backward(self, ground_truth, lr):
        #print("=========================")
        for i in range(len(self.outputs)-1, 0, -1):
            #print("out[{}] = {}".format(i, self.outputs[i]))
            #print("sum_probs = {}".format(numpy.sum(self.outputs[i])))
            #print("t[{}] = {}".format(i, ground_truth))
            if i == len(self.outputs)-1:

                delta = (self.outputs[i] - ground_truth)# * numpy.apply_along_axis(d_softmax, 1, self.inputs[i-1])
                #print("input = {}".format(self.inputs[i-1].shape))
                #print("derivative = {}".format(numpy.apply_along_axis(d_softmax, 1, self.inputs[i-1]).shape))
                #print("error[{}] = {}".format(i-1, (self.outputs[i] - ground_truth).shape))
                #delta = numpy.sum(delta, axis=0)
                #input()
                #print("inputs[{}] = {}".format(i, self.inputs[i].shape))
                #error = error.reshape((error.shape[0], 1))
                #print("delta sum: {}".format(delta.shape))
                
                #derivative = d_softmax(self.inputs[i])
                #derivative = derivative.reshape((derivative.shape[0], 1))
                #print("derivative[{}] = {}".format(i, derivative))
                #delta =  numpy.multiply(error, derivative)#derivative#
                #print("delta[{}] = {}".format(i, delta.shape))
                #print("outputs[{}] = {}".format(i-1, self.outputs[i-1].shape))
                self.deltas.append(delta)
                gradient = numpy.dot(delta.T, self.outputs[i-1])
                #print("gradient[{}] = {}".format(i, gradient.shape))
                #print("gradient_warehouse[{}] = {}".format(i-1, self.gradients[i-1].shape))
                self.gradients[i-1] += gradient
                #print("gradient[{}] = {}".format(i-1, self.gradients[i-1]))
                #input()
                

            else:
                #print("-------------------------")
                #print("weights[{}] = {}".format(i-1, self.weights[i].shape))
                #print("delta[-1] = {}".format(self.deltas[-1].shape))
                delta = numpy.dot(self.deltas[-1], self.weights[i]) * numpy.apply_along_axis(d_sigmoid, 1, self.inputs[i-1])
                #print("input[{}] = {}".format(i-1, d_sigmoid(self.inputs[i-1])))
                #print("derivative = {}".format(d_sigmoid(self.inputs[i-1])))
                #print("inputs[{}] = {}".format(i-1, self.inputs[i].shape))
                #derivative = d_sigmoid(self.inputs[i])
                #print("derivative[{}] = {}".format(i, derivative.shape))
                #delta = numpy.multiply(d, derivative.T)
                self.deltas.append(delta)
                #print("delta[{}] = {}".format(i, delta.shape))
                #print("outputs[{}] = {}".format(i-1, self.outputs[i-1].shape))
                gradient = numpy.dot(delta.T, self.outputs[i-1])
                #print("Sample grad==============\n{}".format(gradient))
                self.gradients[i-1] += gradient
                #print("Sample grad==============\n{}".format(gradient))
                #input()

        self.inputs = []
        self.outputs = []
        self.deltas = []

    def update_weights(self, lr):
        #print("Weights before======================\n{}\n".format(self.weights))
        for i in range(len(self.weights)):
            #print("Gradient[{}] \n{}".format(i, self.gradients[i]))
            #input()
            self.weights[i] += -lr * self.gradients[i]

        self.reset_gradients()
        #print("Weights after======================\n{}".format(self.weights))

    def predict(self, element):
        prediction = self.forward(element)
        self.inputs = []
        self.outputs = []
        return prediction
            

def get_batch(idx, size):
    if idx+size < len(DATA):
        return DATA[idx:idx+size], LABELS[idx:idx+size]
    else:
        return DATA[idx:], LABELS[idx:]

def get_batch_test(idx, size):
    if idx+size < len(TEST_DATA):
        return TEST_DATA[idx:idx+size], TEST_LABELS[idx:idx+size]
    else:
        return TEST_DATA[idx:], TEST_LABELS[idx:]

def load_data(train_samples, train_labels, test_samples, test_labels):
    print("Loading files...")
    train = []
    test = []
    with open(train_samples, 'r') as f: 
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break
            line = line.split(",")
            line = [int(x)/255.0 for x in line]
            line.append(1.0)
            #print(line)
            train.append(line)
    f.close()
    with open(train_labels, 'r') as f:
        i = 0   
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break
            line = int(line)
            train[i].append(line)
            i += 1
    f.close()

    with open(test_samples, 'r') as f:  
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break
            line = line.split(",")
            line = [int(x)/255.0 for x in line]
            line.append(1.0)
            test.append(line)
    f.close()

    with open(test_labels, 'r') as f:
        i = 0  
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break
            line = int(line)
            test[i].append(line)
            i += 1
    f.close()

    random.shuffle(train)
    #random.shuffle(test)
    train_s = [numpy.array(item[:-1]) for item in train]
    train_l = [item[-1] for item in train]
    test_s = [numpy.array(item[:-1]) for item in test]
    test_l = [item[-1] for item in test]
    print("Files loaded!")
    return train_s[:10000], train_l[:10000], test_s, test_l

parser = argparse.ArgumentParser()
parser.add_argument("train_s")
parser.add_argument("train_l")
parser.add_argument("test_s")
parser.add_argument("test_l")
args = parser.parse_args()

DATA, LABELS, TEST_DATA, TEST_LABELS = load_data(args.train_s, args.train_l, args.test_s, args.test_l)
#DATA, LABELS = [[1.0/5,2.0/5,3.0/5,4.0/5,5.0/5, 1.0], [-1.0/5,-2.0/5,-3.0/5,-4.0/5,-5.0/5, 1.0]], [numpy.array([1,0]).reshape(2,1), numpy.array([0,1]).reshape(2,1)]
print("Lenght training: {}\nLenght test: {}".format(len(DATA), len(TEST_DATA)))
N_EPOCHS = 350 #400
BATCH_SIZE = 64
LR = 0.05
ONE_HOT ={0:numpy.array([1,0,0,0,0,0,0,0,0,0]).reshape(10,1), 1:numpy.array([0,1,0,0,0,0,0,0,0,0]).reshape(10,1),
          2:numpy.array([0,0,1,0,0,0,0,0,0,0]).reshape(10,1), 3:numpy.array([0,0,0,1,0,0,0,0,0,0]).reshape(10,1),
          4:numpy.array([0,0,0,0,1,0,0,0,0,0]).reshape(10,1), 5:numpy.array([0,0,0,0,0,1,0,0,0,0]).reshape(10,1),
          6:numpy.array([0,0,0,0,0,0,1,0,0,0]).reshape(10,1), 7:numpy.array([0,0,0,0,0,0,0,1,0,0]).reshape(10,1),
          8:numpy.array([0,0,0,0,0,0,0,0,1,0]).reshape(10,1), 9:numpy.array([0,0,0,0,0,0,0,0,0,1]).reshape(10,1)}

net = Net([784,512,256,128,64,10])
#net = Net([5,4,3,2])
for i in range(N_EPOCHS):
    idx = 0
    end = False
    batch_loss = 0
    while not end:
        if idx >= len(DATA):
            end = True
        batch, labels = get_batch(idx, BATCH_SIZE)
        idx += BATCH_SIZE
        if len(batch) == 0:
            break
        batch = numpy.array(batch)
        labels = [ONE_HOT[x] for x in labels]
        labels = numpy.array(labels)
        labels = labels.squeeze()
        #print("Batch size => {}\nLabels size => {}".format(batch.shape, labels.shape))
        
        pred = net.forward(batch)
        #print(pred)

        for it in range(pred.shape[0]):
            batch_loss += -numpy.sum(numpy.dot(labels[it].T, numpy.log(pred[it]+0.000001)))
        #batch_loss += -numpy.sum(numpy.dot(labels[j].T, numpy.log(pred)))
        #print("Inputs/Outputs = {}/{}".format(net.inputs, len(net.outputs)))
        
        net.backward(labels, LR)
        net.update_weights(LR)
        #if len(batch) > 0:
            #print("TEST ACC. @ EPOCH {}, BATCH {} ===> {}".format(i, idx//BATCH_SIZE, correct_batch/len(batch)))
    print("LOSS. @ EPOCH {} ===> {}".format(i, batch_loss))
    
    """
    if i > 200:
        LR = 0.01
    elif i > 300:
        LR = 0.007
    
    elif i > 700:
        LR = 0.001
    """
    """
    correct = 0
    for j, element in enumerate(TEST_DATA):
        prediction = net.predict(element)
        #max_element = max(prediction)
        p_label = numpy.argmax(prediction)
        print("{}-{}".format(prediction, TEST_LABELS[j]))
        if p_label == TEST_LABELS[j]:
            correct += 1

    print("TEST ACC. @ EPOCH {} ===> {}".format(i, correct/len(TEST_DATA)))
    """
#print("TRAINING IS CORRECT")
correct = 0
idx = 0
end = False
while not end:
    if idx >= len(TEST_DATA):
        end = True
    batch, labels = get_batch_test(idx, BATCH_SIZE)
    idx += BATCH_SIZE
    if len(batch) == 0:
        break
    batch = numpy.array(batch)
    #print("Batch size => {}".format(batch.shape))
    
    pred = net.forward(batch)
    for i in range(pred.shape[0]):
        if numpy.argmax(pred[i]) == labels[i]:
            correct += 1
print("TEST ACCURACY ===> {}".format(correct/len(DATA)))

correct = 0
idx = 0
end = False
file_write = open("test_predictions.csv", "w")
write_buffer = ""
while not end:
    if idx >= len(TEST_DATA):
        end = True
    batch, labels= get_batch_test(idx, BATCH_SIZE)
    idx += BATCH_SIZE
    if len(batch) == 0:
        break
    batch = numpy.array(batch)
    #print("Batch size => {}".format(batch.shape))
    
    pred = net.forward(batch)
    for i in range(pred.shape[0]):
        write_buffer += str(numpy.argmax(pred[i])) + "\n"

write_buffer = write_buffer[:-1]
#print(write_buffer)
file_write.write(write_buffer)
file_write.close()
#print("TEST ACCURACY ===> {}".format(correct/len(TEST_DATA)))
#error = -numpy.sum(CROSSS_ENTROPY(prediction, [1,0]))
#print(numpy.array([1,0]).shape)
#net.backward(numpy.array([1,0]).reshape((2,1)))
#net.update_weights(0.1)
#print(error)




