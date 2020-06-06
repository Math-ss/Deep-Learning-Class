/*
"BackPropagationLearning.h"

Be careful, the softmax mode isn't good !!!!

Created by Math's
*/

#pragma once

#include <vector>
#include <string>
#include "DeepLearningNetwork.h"

/*
Network with backpropagation
*/
class BackPropagationLearning : public DeepLearningNetwork
{
protected:
	std::vector<std::vector<float>> m_delta;
	/*
	Indicates in each under data vector we are for the updateParameters() method
	*/
	int m_trainingProgression;
	
public:
	BackPropagationLearning(int inputLayer, int outputLayer, std::vector<int> &hiddenLayer, std::string &f);

	/*
	Call computePrediction() then updateParameters()
		*repetion -> indicates the absolute number of calls, but the input change at each call (depends on m_data.size())
	*/
	virtual void training(int repetition = 1, float learningRate = 0.5) override;
	virtual void updateParameters(float learningRate = 0.5) override;

	/*
	Gradient descent implementation
	*/
};
