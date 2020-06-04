#pragma once
#include "AI_Interface.h"
#include <string>

class DeepLearningNetwork :
	public AI_Interface
{
protected:
	std::vector<std::vector<std::vector<float>>> m_weights;
	std::vector< std::vector<float>> m_biais;
	std::vector< std::vector<float>> m_perceptrons;
	int m_nbInput;
	int m_nbOutput;

	std::string m_fLogi;

public:
	DeepLearningNetwork(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, std::string& f);
	virtual void computePrediction(std::vector<float>& input) override;

	static float F_Sigmoide(float value);
	static float F_TangenteHyper(float value);
	static void F_Softmax(std::vector<float> &value);

	static float F_Derivative_Sigmoide(float value);

	virtual bool saveWeights(std::string path);
};

