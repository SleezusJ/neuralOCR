//multilevel neural net framework
//you need to fill in the getRawPrediction and train functions

public class Perceptron
{
    private static final double ALPHA=0.05;
    private static final double NOISEMAX=0.4;

    //weights from hidden to output layers
    double[] outputweight;
    //weights from input to hidden layers
    double[][] hiddenweight;

    //temporary space for caching hidden layer values
    double[] hidden;

    //number of nodes in input and hidden layers
    int size;

    //constructor.  Called with the number of inputs:  new Perceptron(3,2) makes a three input, two output perceptron.
    Perceptron(int size)
    {
        this.size=size;
        //make an array of weights from each hidden, plus a bias, to each output node
        outputweight=new double[size+1];
        //make a 2D array of weights from each input, plus a bias, to each hidden node
        hiddenweight=new double[size][size+1];
        for(int i=0; i<size+1; i++)
            outputweight[i]=Math.random()*NOISEMAX-NOISEMAX/2;
        for(int i=0; i<size; i++)
            for(int j=0; j<size+1; j++)
                hiddenweight[i][j]=Math.random()*NOISEMAX-NOISEMAX/2;
        //create the array for caching, but don't bother initializing it
        hidden=new double[size];
    }

    //returns whether the raw prediction is a 1 or 0
    int getPrediction(int[] inputs)
    {
        return getRawPrediction(inputs)>=0.5? 1:0;
    }

    //takes an array of inputs in range 0 to 1, feeds them to the perceptron, saves a guess in range 0 to 1 in array "outputs"
    double getRawPrediction(int[] inputs)
    {


        //TODO:
        //1. rescale the inputs from -1 to 1 and copy them to array inputs
        for(int i = 0; i < inputs.length; i++){
            if(inputs[i] == 0){
                inputs[i] = -1;
            }else{
                inputs[i] = 1;
            }
        }
        //2. compute dot product of inputs times weights for each hidden.  do sigmoid of total and save it in array hidden
        double total=0;
        for(int h=0; h< hidden.length; h++){
            for(int i=0; i< size; i++) {
                total += inputs[i] * hiddenweight[h][i];
            }
            total+= hiddenweight[h][size];
            hidden[h] = sigmoid(total);
        }
        //3. compute dot product of hidden times weights for each output.  do sigmoid and return it
        double outTotal = 0;
        for(int o =0; o < hidden.length; o++){
            outTotal += hidden[o] * outputweight[o];
        }

        return sigmoid(outTotal);
    }


    //this trains the perceptron on an array of inputs (1/0) and desired outputs (1/0)
//the weights are adjusted and errors are saved in array "error".  return TRUE if training is done
    boolean train(int[] inputs, int want)
    {
        float[] errors = new float[size];

        //TODO:
        //1. call getPrediction on inputs.  this will put values in hidden and outputs that we can use for training
        float predicted = getPrediction(inputs);
        //2. compute output error for each output and save it in "errors":  error = desired-predicted
        float error = want - predicted;
        //3. compute output training error for each output node:  outtrainerror = error * predicted * (1-predicted)
        float outtrainerror = error * predicted * (1-predicted);
        //4. compute hidden error for each hidden node:  hiddenerror = sum of (outtrainerror * output weight) over all outputs
        double [] hiddenerror = new double[size];
        for(int i =0; i<size; i++ ){
            hiddenerror[i] = outtrainerror * outputweight[i];
        }
        //5. for each hidden node, apply output training error to weights:  outputweight += alpha * outtrainerror * hidden-value
        //don't forget to train the bias weight.  it has a hidden-value of 1
        for(int j=0; j<hidden.length; j++){
            outputweight[j] += ALPHA * outtrainerror * hidden[j];
        }
        outputweight[size] += ALPHA * outtrainerror * hidden[size];
        //6. over each input, compute hidden training error: hiddentrainerror = hiddenerror * hidden-value * (1-hidden-value)
        double hiddentrainerrors[] = new double[size];
        for(int k=0;k<inputs.length;k++){
            hiddentrainerrors[k] = hiddenerror[k] * hidden[k] * (1-hidden[k]);
        }
        //7. apply that error to the input weight: hiddenweight += hiddentrainingerror * inputvalue * (1-inputvalue)
        for(int l=0;l<hidden.length;l++){
            for(???){
                hiddenweight[l][???] += hiddentrainerrors[l] * inputs[l] * (1-inputs[l]);
            }
        }

        //8. go through all the errors in the array and keep track of the maximum.  if the max error is below some threshold (say 0.1), return TRUE. (else FALSE)

        return false;	//replace this line
    }

    //implements the threshold function 1/(1+e^-x)
//this is mathematically close to the >=0 threshold we use in the single layer perceptron, but is differentiable
    static double sigmoid(double x)
    {
        return 1.0/(1.0+Math.pow(2.71828,-x));
    }
}