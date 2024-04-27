import fs from 'fs';
import * as thedata from "./mnist_train.json" assert { type: "json" };
import * as thedata_test from "./mnist_test.json" assert { type: "json" };

// Normalise Data
const data = normalizeData(thedata);
const data_test = normalizeData(thedata_test);

// Variables
const learningRate = 0.05;
var weights = [];
var bias = [];
var output = [];
var actual = [];
var activations = [];
var sigmoid_derivatives = [];
const inputSize = 784;
const layers = [128, 64, 10];
const epochs = 5;

// labels
const labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
[weights, bias] = getWeightsAndBias(inputSize, layers);


// Training of the model
for (let i = 0; i < epochs; i++) {
    console.log("Epoch: ", i);
    train(data, weights, bias);
    console.log(test(data_test, weights, bias), "accuracy");
}

// write weights and bias to file
writeToFile(weights, bias);


// ----------------------------------------------
// Functions are below
// ----------------------------------------------

// train
function train(data, weights, bias) {
    for (let i = 0; i < data.length; i++) {
        // console.log(i);
        [activations, sigmoid_derivatives, output, actual] = forwardPropagation(data[i], weights, bias);
        backPropagation(data[i].image, weights, bias, activations, sigmoid_derivatives, actual);
    }
}


// test
function test(data, weights, bias) {
    let correct = 0;
    for (let i = 0; i < data.length; i++) {
        [activations, sigmoid_derivatives, output, actual] = forwardPropagation(data[i], weights, bias);
        const outputIndex = output.indexOf(Math.max(...output));
        const actualIndex = actual.indexOf(1);
        if (outputIndex === actualIndex) {
            correct++;
        }
    }
    return correct / data.length;
}

// back propagation
function backPropagation(data, weights, bias, activations, sigmoid_derivatives, actual) {

    let das = Array.from({ length: layers.length }, () => 0);

    das[layers.length - 1] = multiplyArrayStatic(2, differenceArray(activations[layers.length - 1], actual));

    for (let i = layers.length - 1; i >= 0; i--) {
        const al = activations[i];
        const al1 = i == 0 ? data : activations[i - 1];

        const dzl_dwl = al1;
        const dal_dzl = sigmoid_derivatives[i];

        // const dC_dal = multiplyArrayStatic(2, differenceArray(al, actual));

        const dC_dal = das[i];

        const dC_dbl = multiplyArrayDynamic(dal_dzl, dC_dal);

        das[i - 1] = matrixDotProduct([dC_dbl], weights[i])[0];

        const dC_dwl = dotProduct(dzl_dwl, dC_dbl);

        const updated_weights = updateWeights(weights[i], dC_dwl);
        const updated_bias = updateBias(bias[i], dC_dbl);

        weights[i] = updated_weights;
        bias[i] = updated_bias;
    }
}

// Sigmoid function
function sigmoid(x) { 
    return 1 / (1 + Math.exp(-x)) 
}

// get weights based on input and  size of layer
function getWeights(input, size) {
    let x = [];
    for (let i = 0; i < size; i++) {
        x.push(Array.from({ length: input }, () => (Math.random() * 20 - 10) * 0.1));
    }
    return x;    
};


// get bias based on size of layer
function getBias(size) {
    return Array.from({ length: size }, () => (Math.random() * 20 - 10) * 0.1);
};


// get output of layer
function getOutput(input, weights, bias) {
    let output = [];
    let sigmoidDerivatives = []
    for (let i = 0; i < weights.length; i++) {
        let sum = 0;
        for (let j = 0; j < weights[i].length; j++) {
            sum += weights[i][j] * input[j];
        }
        let z = sum + bias[i];
        sigmoidDerivatives.push(sigmoidDerivative(z));
        let a = sigmoid(z)
        output.push(a);
    }
    return [output, sigmoidDerivatives];
};

// one hot encoding
function oneHotEncoding(label) {
    let oneHot = Array.from({ length: 10 }, () => 0);
    oneHot[label] = 1;
    return oneHot;
}

// softmax
function softmax(output) {
    // get max value
    let max = Math.max(...output);
    // return the index of max value
    let index = output.indexOf(max);   
}


// cost function
function costFunction(output, actual) {
    let sum = 0;
    for (let i = 0; i < output.length; i++) {
        sum += Math.pow((actual[i] - output[i]), 2);
    }
    return sum;
}

// sigmoid derivative
function sigmoidDerivative(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// normalize data
function normalizeData(data) {
    let normalizedData = [];
    for (let i = 0; i < data.default.length; i++) {
        let image = data.default[i].image;
        let label = data.default[i].label;
        let normalizedImage = [];
        for (let j = 0; j < image.length; j++) {
            normalizedImage.push(image[j] / 255);
        }
        normalizedData.push({ image: normalizedImage, label: label });
    }
    return normalizedData;
}

// function to difference between array
function differenceArray(a, b) {
    if (a.length !== b.length) {
        return "Array length not equal";
    }
    let arr = [];
    for (let i = 0; i < a.length; i++) {
        arr.push(a[i] - b[i]);
    }
    return arr;
}

// static array multiplication
function multiplyArrayStatic(a, b) {
    let arr = [];
    for (let i = 0; i < b.length; i++) {
        arr.push(a * b[i]);
    }
    return arr;
}

// dynamic array multiplication
function multiplyArrayDynamic(a, b) {
    if (a.length !== b.length) {
        return "Array length not equal";
    }
    let arr = [];
    for (let i = 0; i < a.length; i++) {
        arr.push(a[i] * b[i]);
    }
    return arr;
}


// dot product of two arrays
function dotProduct(a, b) {
    let output = [];
    for (let i = 0; i < b.length; i++) {
        let x = []
        for (let j = 0; j < a.length; j++) {
            x.push(a[j] * b[i]);
        }
        output.push(x);
    }
    return output
}


// update weights
function updateWeights(weights, dC_dwl) {
    let updatedWeights = [];
    for (let i = 0; i < weights.length; i++) {
        let x = [];
        for (let j = 0; j < weights[i].length; j++) {
            x.push(weights[i][j] - learningRate * dC_dwl[i][j]);
        }
        updatedWeights.push(x);
    }
    return updatedWeights;
}

// update bias
function updateBias(bias, dC_dbl) {
    let updatedBias = [];
    for (let i = 0; i < bias.length; i++) {
        updatedBias.push(bias[i] - learningRate * dC_dbl[i]);
    }
    return updatedBias;
}


// get weights and biases from input and layer size
function getWeightsAndBias(input, layers) {
    let weights = []
    let bias = [];

    for (let i = 0; i < layers.length; i++) {
        let ip = i === 0 ? input : layers[i - 1];
        let w = getWeights(ip, layers[i]);
        let b = getBias(layers[i]);
        weights.push(w);
        bias.push(b);
    }
    
    return [weights, bias];
}



// forward propagation
function forwardPropagation(data, weights, bias) {
    let activations = [];
    let sigmoid_derivatives = [];
    
    for (let i = 0; i < weights.length; i++) {
        let act = i == 0 ? data.image : activations[i - 1];
        [activations[i], sigmoid_derivatives[i]] = getOutput(act, weights[i], bias[i]);
    }

    const output = activations[layers.length - 1];
    const actual = oneHotEncoding(data.label);
    return [activations, sigmoid_derivatives, output, actual];
}




// metric dot product

function matrixDotProduct(A, B) {
    // Check if the number of columns in A is the same as the number of rows in B
    if (A[0].length !== B.length) {
        throw new Error("The number of columns in the first matrix must equal the number of rows in the second matrix.");
    }

    let n = A.length;      // Number of rows in the first matrix
    let k = A[0].length;   // Number of columns in the first matrix (or rows in the second)
    let m = B[0].length;   // Number of columns in the second matrix

    // Initialize the result matrix with zeros
    let result = new Array(n);
    for (let i = 0; i < n; i++) {
        result[i] = new Array(m).fill(0);
        for (let j = 0; j < m; j++) {
            for (let r = 0; r < k; r++) {
                result[i][j] += A[i][r] * B[r][j];
            }
        }
    }

    return result;
}

// write weights and bias to file
function writeToFile(weights, bias) {
    let data = JSON.stringify({ weights: weights, bias: bias });
    fs.writeFileSync('weights.json', data);
}
