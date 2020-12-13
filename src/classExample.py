trainingset = [ [[0,0],0], [[0,1],1], [[1,0],1], [[1,1],0]  ]

# net = Perceptron(2)
# net.train([0,1], 1]   #input array and desired output
# net.predict([0,1])    #gives me an output
# net.trainUntilPerfect(trainingset)    #learn everything in that trainingset

NOISEMAGNITUDE = 0.4      #used for initial weights will be -.2 .. .2
ALPHA=0.1                #learning rate - adjustable

import random
import math

class Perceptron:
    #class vars:
    # size: # inputs, # hidden,  single output
    #output weight array  size+1 in length (bias)
    #hidden weight array [size][size+1]
    #temporary storage:  remember  hidden values[size]

    def __init__(self,size):
        self.size=size
        self.oweight=[0]*(size+1)  #output array
        #makes a 2d array for hidden weights
        self.hweight=[0]*size
        for i in range(size):
            self.hweight[i]=[0]*(size+1)

        #make some small random numbers for each weight
        for i in range(size+1):
            #set from -NOISE/2 to NOISE/2
            self.oweight[i]=random.random()*NOISEMAGNITUDE - NOISEMAGNITUDE/2
        for i in range(size):
            for j in range(size+1):
                self.hweight[i][j]=random.random()*NOISEMAGNITUDE - NOISEMAGNITUDE/2

    #this looks like >=0 step function, but is differentiable
    def sigmoid(self,x):
        return 1.0 / (1 + math.exp(-x))

    # called on an array of 0/1 input values.  returns a floating point output
    def predict(self,inputs):
        # first we determine the hidden numbers
        self.hidden=[0]*self.size        # place to store hidden values

        # make a calculation for each hidden node
        for h in range(self.size):
            # go through the inputs, multiplying and summing
            total=0
            for i in range(self.size):
                #recast input[i] as -1/1 instead of 0/1
                if inputs[i]==0:
                    theinput=-1
                else:
                    theinput=1
                total += theinput * self.hweight[h][i]
            #total is weights*inputs
            #don't forget bias
            total += self.hweight[h][self.size]
            #instead of using >=0 threshold, use the sigmoid instead
            self.hidden[h] = self.sigmoid(total)

        #go through each hidden node and compute output
        total=0
        for h in range(self.size):
            total += self.hidden[h] * self.oweight[h]
        total+=self.oweight[self.size]   #bias
        #pass it through threshold to get output
        return self.sigmoid(total)

    # called with an array of inputs and a wanted output
    def train(self,inputs,wanted):
        #train starts by making a prediction
        predicted = self.predict(inputs)
        # now we have a guess (predicted), also hidden guesses

        #first get error
        error = wanted - predicted      #error is floating point

        #differentiate sigmoid(error) to pass back to hiddens
        oerror = error * predicted * (1-predicted)

        #let's compute the error at each hidden level
        herror=[0]*self.size
        for h in range(self.size):
            herror[h]  = self.hidden[h]*(1-self.hidden[h]) * oerror * self.oweight[h]

        #now we know all the errors, let's adjust weights
        #train the outputs
        for h in range(self.size):
            self.oweight[h] += oerror * self.hidden[h] * ALPHA
        #bias:
        self.oweight[self.size] += oerror * ALPHA

        for h in range(self.size):
            for i in range(self.size):
                if inputs[i]==1:
                    theinput=1
                else:
                    theinput=-1
                self.hweight[h][i] += herror[h] * theinput * ALPHA
            #bias
            self.hweight[h][self.size] += herror[h] * ALPHA

        #return the error so I can see how well I learned it
        return error

    #call this with whole trainingset
    def trainUntilPerfect(self, trainingset, cutoff):
        for i in range(cutoff):
            done=True
            for trainingitem in trainingset:
                error=self.train(trainingitem[0],trainingitem[1])
                #print("trianing on ",trainingitem,"error is",error)
                #we've learned it when error get smaller than .1
                if error>0.1 or error<-0.1:
                    #print(error)
                    done=False
            if done:
                return True         #it learned it!
        #hit the cutoff, never learned it
        return False




""""
 for(int h=0; h<size; h++){
                int theInput=0;
                double total = 0;
                for(int i=0; i<size; i++){
                    if(inputs[i] == 0){
                        theInput = -1;
                    }else{
                        theInput = 1;
                    }
                    total += theInput * hiddenweight[h][i];
                }
                total += hiddenweight[h][size];
                hidden[h] = sigmoid(total);
        }

        double total = 0;
        for(int h=0; h<size; h++){
            total+= hidden[h] * outputweight[h];
        }
        total+= outputweight[size];

        return sigmoid(total);


""""