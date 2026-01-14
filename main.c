//basic neural network that predicts sin() of quadrantal angles
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define EPOCHS 1000
#define LEARNING_RATE 0.1
#define NUM_INPUTS 4
#define NUM_HIDDEN 4
#define NUM_OUTPUTS 3


//get a random value between 0 and 1
double getRand()
{
	return (double)rand() / (RAND_MAX + 1.0);
}

//sigmoid and sigmoid derivative
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
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
	//0, 90, 180, 270
	double TRAINING_INPUTS[4][NUM_INPUTS] = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1}
	};
	//0, 1, -1
	double TRAINING_OUTPUTS[4][3] = {
		{1, 0, 0},
		{0, 1, 0},
		{1, 0, 0},
		{0, 0, 1}
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

	//for each input,
	for(int input = 0; input < 4; input++)
	{
		//forward pass
		//hidden layer
		double HIDDEN_INPUTS[NUM_HIDDEN]; //values given to hidden layer
		double HIDDEN_RESULT[NUM_HIDDEN]; //values computed by hidden layer neurons
		//for each hidden neuron,
		for(int i = 0; i < NUM_HIDDEN; i++)
		{
			//for each input neuron,
			double result = 0.f;
			for(int j = 0; j < NUM_INPUTS; j++)
			{
				//get input
				double inputValue = TRAINING_INPUTS[input][j];
				//apply weight
				inputValue *= HIDDEN_WEIGHTS[j][i];
				//add to result
				result += inputValue;
			}

			//apply hidden neuron bias
			result += HIDDEN_BIASES[i];

			//store hidden neuron input
			HIDDEN_INPUTS[i] = result;

			//apply activation to introduce non-linearity
			result = sigmoid(result);

			//store final output
			HIDDEN_RESULT[i] = result;
		}

		//output layer
		double OUTPUT_INPUTS[NUM_OUTPUTS];
		double OUTPUT_RESULT[NUM_OUTPUTS];
		//for each output neuron,
		for(int i = 0; i < NUM_OUTPUTS; i++)
		{
			double result = 0.f;
			//for each hidden neuron,
			for(int j = 0; j < NUM_HIDDEN; j++)
			{
				//get input (from hidden neuron)
				double inputValue = HIDDEN_RESULT[j];

				//apply weight
				inputValue *= OUTPUT_WEIGHTS[j][i];

				//add to result
				result += inputValue;
			}

			//apply output neuron bias
			result += OUTPUT_BIASES[i];

			//store output neuron input
			OUTPUT_INPUTS[i] = result;

			//apply activation to introduce non-linearity
			result = sigmoid(result);

			//store final result
			OUTPUT_RESULT[i] = result;
		}

		//loss calculation
		double totalLoss;
		double OUTPUT_LOSS[NUM_OUTPUTS];
		//for each output neuron,
		for(int i = 0; i < NUM_OUTPUTS; i++)
		{
			//Mean Squared Error (multiplied by 0.5 for simpler derivative)
			double loss = 0.5 *
				(OUTPUT_RESULT[i] - TRAINING_OUTPUTS[input][i]) *
				(OUTPUT_RESULT[i] - TRAINING_OUTPUTS[input][i]);

			//store loss
			OUTPUT_LOSS[i] = loss;

			//accumulate total loss
			totalLoss += loss;
		}

		//back propagation

		//output error

		//bias -= (learning rate)(change required)
		double OUTPUT_ERRORS[NUM_OUTPUTS];
		//for each output neuron,
		for(int i = 0; i < NUM_OUTPUTS; i++)
		{
			//change required = partial d(loss)/d(input)
			// =
			//partial d(loss)/d(output) *
			//partial d(output)/d(input)
			// =
			//(network output - expected output) *
			// sigmoid_derivative(neuron input)
			double error =
				(OUTPUT_RESULT[i] - TRAINING_OUTPUTS[input][i]) *
				dSigmoid(OUTPUT_INPUTS[i]);

			//store error
			OUTPUT_ERRORS[i] = error;
		}

		//hidden error
		double HIDDEN_ERRORS[NUM_HIDDEN];
		//for each hidden neuron,
		for(int i = 0; i < NUM_HIDDEN; i++)
		{
			double error = 0.f;
			//for each output neuron,
			for(int j = 0; j < NUM_OUTPUTS; j++)
			{
				error += OUTPUT_ERRORS[j] * OUTPUT_WEIGHTS[i][j];
			}
			error *= dSigmoid(HIDDEN_INPUTS[i]);

			//store error
			HIDDEN_ERRORS[i] = error;
		}

		//apply correction
		//output layer
		for(int i = 0; i < NUM_OUTPUTS; i++)
		{
			//adjust biases
			OUTPUT_BIASES[i] -= LEARNING_RATE * OUTPUT_ERRORS[i];
			//adjust weights
			for(int j = 0; j < NUM_HIDDEN; j++)
			{
				OUTPUT_WEIGHTS[j][i] -= LEARNING_RATE * (OUTPUT_ERRORS[i] * OUTPUT_WEIGHTS[j][i]);
			}
		}

		//hidden layer
		for(int i = 0; i < NUM_HIDDEN; i++)
		{
			//adjust biases
			HIDDEN_BIASES[i] -= LEARNING_RATE * HIDDEN_ERRORS[i];
			//adjust weights
			for(int j = 0; j < NUM_INPUTS; j++)
			{
				HIDDEN_WEIGHTS[j][i] -= LEARNING_RATE * (HIDDEN_ERRORS[i] * HIDDEN_WEIGHTS[j][i]);
			}
		}

	}
	return 0;
}
