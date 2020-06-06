/*
"DeepLearningNetwork.h"

Created by Math's
*/

#pragma once
#include "AI_Interface.h"
#include <string>
#include <ctype.h>

/*
Basic interface for a perceptron's network.

-For back propagation see : 
class BackPropagationLearning

-For genetical algorithm see:
(coming soon !!)
*/

class DeepLearningNetwork :
	public AI_Interface
{
protected:
	std::vector<std::vector<std::vector<float>>> m_weights;
	std::vector< std::vector<float>> m_biais;

	/*
	Activation value of each perceptron
	*/
	std::vector<std::vector<float>> m_perceptrons;
	int m_nbInput;
	int m_nbOutput;

	std::string m_fLogi;

public:
	DeepLearningNetwork(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, AIF_Activation FActivation = SIGMOIDE);

	/*
	Make a FeedForward
	*/
	virtual void computePrediction(TrainingParameters *param, std::vector<float> *input) override;

	/*
	Return the last percptron's layer : the Network's output layer. By const pointer
	*/
	virtual std::vector<float> const *getPredictionPtr() const override;
	
	/*
	Return the last percptron's layer : the Network's output layer. Return a copy or a const ref
	*/
	virtual std::vector<float> getPrediction() const override;

	/*
	Usefull non-linear function.
	*/
	static float F_Sigmoide(float value);
	/*
	Usefull non-linear function.
	*/
	static float F_TangenteHyper(float value);

	static float F_Derivative_Sigmoide(float value);

	/*
	Write all wheights in a info.wtw file
		*path -> the path with a / at the end
	*/
	virtual bool saveWeights(std::string path);
};

