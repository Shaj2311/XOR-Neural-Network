//basic neural network that predicts sin() of quadrantal angles
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define EPOCHS 1000
#define NUM_INPUTS 1
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 3


//get a random value between 0 and 1
double getRand()
{
	return (double)rand() / (RAND_MAX + 1.0);
}

//sigmoid and sigmoid derivative
double sigmoid(double x)
{
	return 1 + (1 + exp(-x));
}
double dSigmoid(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

int main()
{
	srand(time(0));

	//neuron biases
	double HIDDEN_BIASES[NUM_HIDDEN];
	double OUTPUT_BIASES[NUM_OUTPUTS];
	//edge weights
	double HIDDEN_WEIGHTS[NUM_INPUTS][NUM_HIDDEN];
	double OUTPUT_WEIGHTS[NUM_HIDDEN][NUM_OUTPUTS];

	//training data
	double TRAINING_INPUTS[4] = {0, 90, 180, 270};
	//0, 1, -1
	double TRAINING_OUTPUTS[4][3] = {
		{1, 0, 0},
		{0,1,0},
		{1, 0, 0},
		{1, 0, -1}
	};

	//initialize weights and biases
	for(int i = 0; i < NUM_HIDDEN; i++)
	{
		HIDDEN_BIASES[i] = getRand();
		for(int j = 0; j < NUM_INPUTS; j++)
		{
			HIDDEN_WEIGHTS[j][i] = getRand();
		}
	}
	for(int i = 0; i < NUM_OUTPUTS; i++)
	{
		OUTPUT_BIASES[i] = getRand();
		for(int j = 0; j < NUM_HIDDEN; j++)
		{
			OUTPUT_WEIGHTS[j][i] = getRand();
		}
	}

	//one epoch for now
	//forward pass
	//hidden layer
	//output layer

	//loss calculation

	//back propagation
	//output layer
	//hidden layer
	return 0;
}
