# Basic-Neural-Network
A simple neural network that predicts XOR (most of the time :) ), written in C
## Network Details
Here is a quick overview of the network's structure and workings:
| Hidden Layers     | 1                             |
| ----------------- | ----------------------------- |
| Input Neurons     | 2                             |
| Hidden Neurons    | 2                             |
| Output Neurons    | 1                             |
| ----------------- | ----------------------------- |
| Initialization    | Random value in range [-1,1)  |
| Activation        | Sigmoid function              |
| Loss Calculation  | MSE                           |
| ----------------- | ----------------------------- |
| Learning Rate     | 0.2                           |
| Training epochs   | 10000                         |
## Try it out
* Clone this project
```
git clone https://github.com/Shaj2311/Basic-Neural-Network
```
* Compile using gcc
```
gcc ./main.c -o main
```
* Execute
```
./main.exe
```
