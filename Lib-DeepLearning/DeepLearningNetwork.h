/*
"DeepLearningNetwork.h"

Created by Math's
*/

#pragma once

#include <string>
#include <ctype.h>
#include "AI_Interface.h"

/*
Basic interface for a perceptron's network.

-For back propagation see : 
class BackPropagationLearning

-For genetical algorithm see:
class GeneticalLearning
*/

class DeepLearningNetwork : public AI_Interface
{
public:

	/*
	If you need to extend this list, create the same enum (same type name) and add your constant at the end. (try to keep the same order)
	Override the necessary functions accordingly and add if necessary the correct activation functions (static).

	DON'T USE INHERITED CONSTRUCTOR(S) TO SET m_FActivation !
	The copy construcor is safe (uses directly the type uint8_t)

	When you are outside the class, always use : YourClass::CONSTANT to avoid issues
	*/
	enum AIF_Activation
	{
		SIGMOIDE,
		TANGEANTE_HYPER
	};

protected:

	std::vector<std::vector<std::vector<float>>> m_weights;
	std::vector< std::vector<float>> m_biais;

	/*
	Activation value of each perceptron
	*/
	std::vector<std::vector<float>> m_perceptrons;

	int m_nbInput;
	int m_nbOutput;

	/*
	Network activation function
	*/
	uint8_t m_FActivation;

public:
	DeepLearningNetwork(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, AIF_Activation FActivation = SIGMOIDE);
	DeepLearningNetwork(DeepLearningNetwork& source);

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
	Write all wheights in a info.wtw file
		*path -> the path with a / at the end
	*/
	virtual bool saveWeights(std::string path);

	/*
	Call the correct activation function (depends on m_FActivation)
	*/
	virtual float F_Activation(float value);

	/*
	Sigmoide activation function
	*/
	static float F_Sigmoide(float value);
	/*
	Hyperbolic tangeant activation function
	*/
	static float F_TangenteHyper(float value);
};

