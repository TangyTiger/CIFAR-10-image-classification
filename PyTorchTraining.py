#from torchvision.datasets import CIFAR10
#from torchvision.transforms import transforms
#from torch.utils.data import DataLoader


#transformations = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])

#batch_size = 10
#number_of_labels = 10

##downloads CIFAR10 train dataset
#train_set = CIFAR10(root="./data", train=True, transform=transformations, download=True)


#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
#print("The number of images in a training set is: ", len(train_loader)*batch_size)


#test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
#print("The number of images in a test set is: ", len(test_loader)*batch_size)

#print("The number of batches per epoch is: ", len(train_loader))
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
print("hi")
# Define a CNN
class Network(nn.Module):
    print("hi2")
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels = 12, out_channels=24, kernel_size = 5, stride =1, padding =1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 5, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(input)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(input)))
        output = F.relu(self.bn5(self.conv5(input)))
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

# Initiate Network
model = Network()

from torch.optim import Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

from torch.autograd import Variable

# function to save model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():

    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            #run the model on the test set to predict labels
            outputs = model(images)

            #the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total+=labels.size(0)
            accuracy += (predicted==labels).sum().item()

    # compute the accuracy over all the test images
    accuracy = (1000 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize
def train(num_epochs):

    best_accuracy = 0.0

    #define your execution device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    #convert model perameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs): # loop over dataset epoch amount of times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            #get input
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            #zero the perameter gradients
            optimizer.zero_grad()

            #predict classes using images from the training set
            outputs = model(images)

            #compute the loss based on models output and real labels
            loss = loss_fn(outputs, labels)

            #backpropogate the loss
            loss.backward()

            #adjust perameters based on the calculated gradients
            optimizer.step()

            #print stats for every 1000 images
            running_loss += loss.item()  #extract the loss value

            if i % 1000 == 999:
                #print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i+1, running_loss/1000))

                #zero the loss
                running_loss = 0.0

        accuracy = testAccuracy() 
        print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # save the model if accuracy is the bst

        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


import matplotlib.pyplot as plt
import numpy as np

#function to show images
def imageShow(img):
    img = img/2 + 0.5   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def testBatch():

    images, labels = next(iter(test_loader))

    imageShow(torchvision.utils.make_grid(images))

    print('Read labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(batch_size)))


    if __name__ == "__main__":

        #build model
        train(5)
        print('Finished Training')

        #test which has preformed well
        testModelAccuracy()

        #load the model created to test accuracy per lavel
        model = Network()
        path = "myFirstModel.pth"
        model.load_state_dict(torch.load(path))

        testBatch()