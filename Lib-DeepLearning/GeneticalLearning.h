/*
"GeneticalLearning.h"

Created by Math's
*/

#pragma once

#include "DeepLearningNetwork.h"

/*
New parameters for genetical learning
*/
struct GeneticalParameters : public TrainingParameters
{
	/*
	Is added to m_fitness when training() is called if end is false
	*/
	int fitnessDiff;
	
	/*
	True if the traning is over (and updateParameters() is called)
	*/
	bool end;

	/*
	Indicates the chances for a given weight to be modified (in range [0.0f, 1.0f]
	*/
	float condition;
	
	GeneticalParameters() : fitnessDiff(0), end(false), condition(0.1f) {}
};

class GeneticalLearning : public DeepLearningNetwork
{
protected:

	/*
	Network's fitness : his score
	*/
	int m_fitness;

public:
	GeneticalLearning(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, AIF_Activation FActivation = SIGMOIDE);
	GeneticalLearning(GeneticalLearning& source);

	/*
	Randomize some wheights (for more details see GeneticalParameters.condition)
		*param must be of type GeneticalParameters
	*/
	virtual void updateParameters(TrainingParameters* param) override;

	/*
	Call updateParameters() if param->end is true, else it just change m_fitness
		*param must be of type GeneticalParameters
	*/
	virtual void training(TrainingParameters* param) override;

	/*
	Add diff to m_fitness
	*/
	virtual void changeFitness(int diff);

	/*
	Set m_fitness to newFit
	*/
	virtual void setFitness(int newFit);

	/*
	Return m_fitness
	*/
	virtual int getFitness();

	/*
	Return a new random wheight (range depends on m_FActivation)
	*/
	virtual float randomWeight();

	/*
	Return a random float in range [min, max]
	*/
	static float RandomFloatInRange(float min, float max);
};

