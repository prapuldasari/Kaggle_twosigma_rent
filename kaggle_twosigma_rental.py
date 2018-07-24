# Import the relevant components
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import cntk as c

# Selecting the right target device when this notebook is being tested:
if 'TEST-DEVICE' in os.environ:
    if os.environ['TEST_DEVICE']=='cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))
    
# Test for CNTK version
if not C.__version__== "2.0":
    raise Exception("This lab is designed to work with cntk 2.0")
    
#Initialization
np.random.seed(0)
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()

#Define the data dimensions
input_dim= 784
num_output_classes= 10


#Data Reading
# Reading a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader( path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field='labels', shape= num_lavel_classes, is_sparse= False)
    featureStream= C.io.StreamDef(field='features', shape= input_dim, is_sparse= False)
    deserailizer= C.io.CTFDeserializer(path, C.io.StreamDefs(labels= labelStream, features= featureStream))
    
    return C.io.MinibatchSource(deserializer, randomize= is_training, max_sweeps= C.io.INFINITELY_REPEAT if is_training else 1)

# Ensuring the training and test data is generated and available for this lab.
# We will search in two locations in the toolkit for the cached MNIST data set.

data_found= False
for data_dir in [os.path.join("..", "Examples", "Image", "Datasets", "MNIST"), os.path.join("data", "MNIST")]:
    train_file= os.path.join(data_dir, "Train-28X28_cntk_text.txt")
    test_file= os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found= True
        break
if not data_found:
    raise ValueError("Please generate the data")

print ("Data directory is {0}".format(data_dir))
    

#Verifying the data dimensions
input= C.input_variable(input_dim)
label= C.input_variable(num_output_classes)

#Creating the model using dense which creates a network

def create_model(features):
    with C.layers.default_options(init= C.glorot_uniform()):
        r= C.layers.Dense(num_output_classes, activate= None)(features)
        return r
# Scale the input to 0-1 range by dividing each pixel by 255.
input_s= input/255
z= create_model(input_s)


#defining loss function and evalustion function for finiding the error rate
loss= C.cross_entropy_with_softmax(z, label)

label_error= C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate=0.2
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner= C.sgd(z.parameters, lr_schedule)
trainer= C.trainer(z, (loss, label_error), [learner])


# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_trainig_progress(trainer, mb, frequency, verbose=1):
    training_loss= "NA"
    eval_loss= "NA"
    
    if mb%frequency == 0:
        training_loss= trainer.previous_minibatch_loss_average
        eval_error= trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1: .4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
    return mb, training_loss, eval_error

# Running the trainer model:
# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

#Creating the reader object
reader_train= create_reader(train_file, True, input_dim, num_output_classes)

#map the data stream to the input and labels
input_map= {label: reader_train.streams.labels, input: reader_train.streams.features}

#running the trainer on and perform model training
training_progress_output_freq= 500

plotdata= {"batchsize":[], "loss":[], "error":[]}
for i in range(0, int(num_minibatches_to_train)):
    #reading a minibatch data
    data= reader_train.next_minibatch(minibatch_size, input_map= input_map)
    trainer.train_minibatch(data)
    batchsize,loss, error= print_trainig_progress(trainer, i, trainig_progress_output_freq,verbose=1)
    
    if not (loss= "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
        
#Computing the moving averages loss to smooth out the noise in SGD
plotdata["avgloss"]= moving_average(plotdata["loss"])
plotdata["avgerror"]= moving_average(plotdata["error"])

#plot the training loss and the training error
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('MiniBatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs Training loss')
plt.show()
plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()



#Evaluating the test set results
reader_test= create_reader(test_file, False, input_dim, num_output_classes)
test_input_map={label: reader_test.streams.labels, input : reader_test_streams.features,}

#Test data for trained model
test_minibatch_size= 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0
for i in range(num_minibatches_to_test):
    data = reader_test.next_minibatch(test_minibatch_size,
                                      input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

out= C.softmax(z)

# Read the data for evaluation
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = {input: reader_eval.streams.features} 

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

img_label = data[label].asarray()
img_data = data[input].asarray()
predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]


# Find the index with the maximum value for both predicted as well as the ground truth
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

print("Label    :", gtlabel[:25])
print("Predicted:", pred)


# Plot a random image
sample_number = 5
plt.imshow(img_data[sample_number].reshape(28,28), cmap="gray_r")
plt.axis('off')

img_gt, img_pred = gtlabel[sample_number], pred[sample_number]
print("Image Label: ", img_pred)