//basic neural network that predicts sin()
#include <math.h>
#define EPOCHS 1000
#define NUM_INPUTS 1
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1

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
	//one epoch for now

	//forward pass
	//hidden layer
	//output layer

	//back propagation
	//output layer
	//hidden layer
	return 0;
}
