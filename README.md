# XOR Neural Network
A simple neural network that predicts XOR (most of the time :D ), written in C
## Network Details
Here is a quick overview of the network's structure and workings:
| Property          | Description                   |
| ----------------- | ----------------------------- |
| Hidden Layers     | 2                             |
| Input Neurons     | 2                             |
| Hidden Neurons    | 2                             |
| Output Neurons    | 1                             |
| Initialization    | Random value in range [-1,1)  |
| Activation        | Sigmoid function              |
| Loss Calculation  | MSE                           |
| Learning Rate     | 0.2                           |
| Training epochs   | 10000                         |
## Try it out
* Clone this project
```
git clone https://github.com/Shaj2311/XOR-Neural-Network.git
```
* Compile using gcc
```
gcc ./main.c -o main
```
* Execute
```
./main.exe
```
