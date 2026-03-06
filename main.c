//basic neural network that predicts xor
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define EPOCHS 10000
#define LEARNING_RATE 0.2
#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1


//get a random value between 0 and 1
double getRand()
{
	double random = (double)rand() / (RAND_MAX + 1.0);
	return (random * 2) - 1;
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
	double TRAINING_INPUTS[4][NUM_INPUTS] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	double TRAINING_OUTPUTS[4][1] = {
		{0},
		{1},
		{1},
		{0}
	};

	//keep retraining the network until it achieves an acceptable accuracy
	while(1)
	{
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

		double totalLoss = 0.f;

		//epoch loop
		for(int epoch = 0; epoch < EPOCHS; epoch++)
		{
			//for each input,
			for(int inputLoop = 0; inputLoop < 4; inputLoop++)
			{
				//training input set is chosen sequentially if it's the last epoch,
				//otherwise chosen randomly
				int input = (epoch == EPOCHS - 1 ? inputLoop : rand() % 4);

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
				totalLoss = 0.f;
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
				if(epoch % 250 == 0)
				{
					printf("\033[H\033[2J");
					printf("Training network...\n");
					printf("Progress: %.2f%%\n", ((double)epoch / EPOCHS) * 100);
					printf("Loss: %f\n", totalLoss);
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
						OUTPUT_WEIGHTS[j][i] -= LEARNING_RATE * (OUTPUT_ERRORS[i] * HIDDEN_RESULT[j]);
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
						HIDDEN_WEIGHTS[j][i] -= LEARNING_RATE * (HIDDEN_ERRORS[i] * TRAINING_INPUTS[input][j]);
					}
				}

			}
		}

		//retrain network if needed
		if(totalLoss >= 0.001)
		{
			printf("Network accuracy too low, press any key to retry\n");
			_getch();
			continue;
		}

		break;
	}


	printf("\nNetwork trained successfully\n");
	printf("Test the network on 0 and 1 values\n");
	printf("Input -1 to quit at any time\n\n");

	//Testing network on user input
	while(1)
	{
		//get user input
		double TESTING_INPUTS[2];

		printf("Enter input 1: ");
		scanf_s("%lf", &TESTING_INPUTS[0]);

		if(TESTING_INPUTS[0] == -1)
			break;
		if(TESTING_INPUTS[0] != 0 && TESTING_INPUTS[0] != 1)
		{
			printf("Invalid input, try again\n\n");
			continue;
		}

		printf("Enter input 2: ");
		scanf_s("%lf", &TESTING_INPUTS[1]);

		if(TESTING_INPUTS[1] == -1)
			break;
		if(TESTING_INPUTS[1] != 0 && TESTING_INPUTS[1] != 1)
		{
			printf("Invalid input, try again\n");
			continue;
		}

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
				double inputValue = TESTING_INPUTS[j];
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

		//print prediction
		printf("Predicted output: %lf (%d)\n\n", OUTPUT_RESULT[0], OUTPUT_RESULT[0] < 0.5 ? 0 : 1);
	}

	return 0;
}
