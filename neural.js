import fs from 'fs';
import * as thedata from "./mnist_train.json" assert { type: "json" };
import * as thedata_test from "./mnist_test.json" assert { type: "json" };


import { train,
    test,
    backPropagation,
    sigmoid,
    getWeights,
    getBias,
    getOutput,
    oneHotEncoding,
    softmax,
    costFunction,
    sigmoidDerivative,
    normalizeData,
    differenceArray,
    multiplyArrayStatic,
    multiplyArrayDynamic,
    dotProduct,
    updateWeights,
    updateBias,
    getWeightsAndBias,
    forwardPropagation,
    matrixDotProduct
} from './algo.js';


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
const layers = [16, 16, 10];
const epochs = 5;

// labels
const labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
[weights, bias] = getWeightsAndBias(inputSize, layers);


// Training of the model
for (let i = 0; i < epochs; i++) {
    console.log("Epoch: ", i);
    train(data, weights, bias, learningRate);
    console.log(test(data_test, weights, bias), "accuracy");
}

// write weights and bias to file
writeToFile(weights, bias);

// write weights and bias to file
function writeToFile(weights, bias) {
    let data = JSON.stringify({ weights: weights, bias: bias });
    fs.writeFileSync('weights.json', data);
}
