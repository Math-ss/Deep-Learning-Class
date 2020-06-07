/*
"BackPropagationLearning.h"

Created by Math's
*/

#pragma once

#include <vector>
#include <string>
#include "DeepLearningNetwork.h"

/*
New parameter for backpropagation
*/
struct BackPropagationParameters : public TrainingParameters
{
	/*
	Learning rate of backpropagation
	*/
	float learningRate;

	BackPropagationParameters() : learningRate(0.5) {}
};

/*
Perceptron network with backpropagation
*/
class BackPropagationLearning : public DeepLearningNetwork
{
protected:
	std::vector<std::vector<float>> m_delta;

	/*
	Specifies which subtables of m_data is currently being used
	*/
	int m_trainingProgression;
	
public:
	BackPropagationLearning(int inputLayer, int outputLayer, std::vector<int> &hiddenLayer, AIF_Activation FActivation = SIGMOIDE);
	BackPropagationLearning(BackPropagationLearning& source);

	/*
	Call computePrediction() then updateParameters() (by updating m_trainingProgression)
		*param must be of type BackPropagationParameters
	*/
	virtual void training(TrainingParameters *param) override;

	/*
	Gradient descent implementation
		*param must be of type BackPropagationParameters
	*/
	virtual void updateParameters(TrainingParameters* param) override;

	/*
	Call correct derivative function
	*/
	virtual float F_Derivative_Activation(float value);

	/*
	Derivative(for gradient descent)
		*Be carefull value is y and not x
	*/
	static float F_Derivative_Sigmoide(float value);
	/*
	Derivative(for gradient descent)
		*Be carefull value is y and not x
	*/
	static float F_Derivative_TangenteHyper(float value);
};
