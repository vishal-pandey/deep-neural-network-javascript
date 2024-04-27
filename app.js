import { forwardPropagation, softmax, normalizeData } from './algo.js';
// var data = {"image": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 125, 171, 255, 255, 150, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 253, 253, 253, 218, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 213, 142, 176, 253, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 250, 253, 210, 32, 12, 0, 6, 206, 253, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 251, 210, 25, 0, 0, 0, 122, 248, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 18, 0, 0, 0, 0, 209, 253, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 117, 247, 253, 198, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 247, 253, 231, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 246, 253, 159, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 234, 253, 233, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 248, 253, 189, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 200, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 253, 173, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 43, 20, 20, 20, 20, 5, 0, 5, 20, 20, 37, 150, 150, 150, 147, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 253, 253, 253, 253, 253, 168, 143, 166, 253, 253, 253, 253, 253, 253, 253, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 249, 247, 247, 169, 117, 117, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 123, 123, 123, 166, 253, 253, 253, 155, 123, 123, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "label": 2}

var predictedValue = NaN;

var data =  Array.from({ length: 784 }, () => 0);

window.onload = async function () {
    let url = window.location.href+"weights.json";
    let weightsAndBias = await fetchWeightsAndBias(url);
    let weights = weightsAndBias.weights;
    let bias = weightsAndBias.bias;

    fillBoard(weights, bias);

    document.querySelector(".loading").style.display = "none";

    document.querySelector(".reset").addEventListener("click", function () {
        data = Array.from({ length: 784 }, () => 0);
        document.querySelector(".output").innerHTML = "";
        fillBoard(weights, bias);
    })

}



async function fetchWeightsAndBias(url) {
    return new Promise((resolve, reject) => {
        fetch(url)
            .then(response => response.json())
            .then(data => {
                resolve(data);
            })
            .catch(err => reject(err));
    })
}

function predict(image, weights, bias) {
    var data = { image: image, label: 0 };

    var thedata = { 
        default: [ data ]
    }
    thedata = normalizeData(thedata);

    data = thedata[0]

    var activations, sigmoid_derivatives, output, actual;
    [activations, sigmoid_derivatives, output, actual] = forwardPropagation(data, weights, bias);

    return softmax(output)
}



function fillBoard(weights, bias) {
    var board = document.querySelector(".board");
    let cellsCount = 28 * 28;

    board.innerHTML = "";

    for (let i = 0; i < cellsCount; i++) {
        let cell = document.createElement("span");
        cell.classList.add("cell");
        cell.id = i;
        cell.addEventListener("mouseover", function (event) {
            if (event.buttons == 1) {
                paintBoard(event.target.id, weights, bias);
            }
        });

        cell.addEventListener("touchmove", function (event) {
            console.log(event)
            let cellId = document.elementFromPoint(event.touches[0].clientX, event.touches[0].clientY).id

            paintBoard(cellId, weights, bias);
        });
        
        board.appendChild(cell);
    }

    

}


// paint the board and predict the value
function paintBoard(cellId, weights, bias) {

    data[cellId] = 255;

    let cellId_upped = parseInt(cellId) - 28;
    let cellId_down = parseInt(cellId) + 28;

    let cellId_left = parseInt(cellId) - 1;
    let cellId_right = parseInt(cellId) + 1;

    let cellId_top_left = parseInt(cellId) - 29;
    let cellId_top_right = parseInt(cellId) - 27;

    let cellId_bottom_left = parseInt(cellId) + 27;
    let cellId_bottom_right = parseInt(cellId) + 29;

    document.getElementById(cellId).style.backgroundColor = "black";
    
    
    if (cellId_upped >= 0) {
        if(data[cellId_upped] != 255) {
            data[cellId_upped] = 120;
            document.getElementById(cellId_upped).style.backgroundColor = "grey";
        }
    }
    if (cellId_down < 784) {
        if(data[cellId_down] != 255) {
            data[cellId_down] = 120;
            document.getElementById(cellId_down).style.backgroundColor = "grey";
        }
    }
    if (cellId_left >= 0) {
        if(data[cellId_left] != 255) {
            data[cellId_left] = 120;
            document.getElementById(cellId_left).style.backgroundColor = "grey";
        }
    }
    if (cellId_right < 784) {
        if(data[cellId_right] != 255) {
            data[cellId_right] = 120;
            document.getElementById(cellId_right).style.backgroundColor = "grey";
        }
    }
    if (cellId_top_left >= 0) {
        if(data[cellId_top_left] != 255 && data[cellId_top_left] != 120) {
            data[cellId_top_left] = 50;
            document.getElementById(cellId_top_left).style.backgroundColor = "lightgrey";
        }
    }
    if (cellId_top_right >= 0) {
        if(data[cellId_top_right] != 255 && data[cellId_top_right] != 120) {
            data[cellId_top_right] = 50;
            document.getElementById(cellId_top_right).style.backgroundColor = "lightgrey";
        }
    }
    if (cellId_bottom_left < 784) {
        if(data[cellId_bottom_left] != 255 && data[cellId_bottom_left] != 120) {
            data[cellId_bottom_left] = 50;
            document.getElementById(cellId_bottom_left).style.backgroundColor = "lightgrey";
        }
    }
    if (cellId_bottom_right < 784) {
        if(data[cellId_bottom_right] != 255 && data[cellId_bottom_right] != 120) {
            data[cellId_bottom_right] = 50;
            document.getElementById(cellId_bottom_right).style.backgroundColor = "lightgrey";
        }
    }

    predictedValue = predict(data, weights, bias);

    document.querySelector(".output").innerHTML = predictedValue;

}

// console.log(sigmoid(0.5));