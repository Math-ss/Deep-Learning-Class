/*
deepBackPropagationLearning.h

int m_nb is for debuging

Created by
MATHGIQUE
*/

#pragma once

#include <vector>
#include <string>
#include "DeepLearningNetwork.h"

class BackPropagationLearning : public DeepLearningNetwork
{
protected:
	std::vector<std::vector<float>> m_delta;
	int m_trainingProgression;
	
public:
	BackPropagationLearning(int inputLayer, int outputLayer, std::vector<int> &hiddenLayer, std::string &f);

	virtual void training(int repetition = 1, float learningRate = 0.5) override;
	virtual void updateParameters(float learningRate = 0.5) override;

	void runLearning(std::vector<std::vector<double>> *test, std::vector<std::vector<double>> *resultDesired);
	void runPrediction(std::vector<double> *data);
	bool saveWeights(std::string path);
};
