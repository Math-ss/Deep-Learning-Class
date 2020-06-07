/*
"GeneticalLearning.h"

Created by Math's
*/

#include "GeneticalLearning.h"

using namespace std;

GeneticalLearning::GeneticalLearning(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, AIF_Activation FActivation)
	:DeepLearningNetwork(inputLayer, outputLayer, hiddenLayer), m_fitness(0)
{
	m_FActivation = FActivation;
}

GeneticalLearning::GeneticalLearning(GeneticalLearning& source)
	:DeepLearningNetwork(source), m_fitness(0)
{}

void GeneticalLearning::updateParameters(TrainingParameters* param)
{
	GeneticalParameters* GeneticalParam = static_cast<GeneticalParameters*>(param);

	for(int i = 0; i < m_weights.size(); i++)
		for (int j = 0; j < m_weights[i].size(); j++)
			for (int k = 0; k < m_weights[i][j].size(); k++)
			{
				if (GeneticalParam->condition >= RandomFloatInRange(0.0f, 1.0f))
					m_weights[i][j][k] = randomWeight();
			}

}

void GeneticalLearning::training(TrainingParameters* param)
{
	GeneticalParameters* GeneticalParam = static_cast<GeneticalParameters*>(param);

	if (GeneticalParam->end)
		return updateParameters(param);

	changeFitness(GeneticalParam->fitnessDiff);
}

void GeneticalLearning::changeFitness(int diff)
{
	m_fitness += diff;
}

void GeneticalLearning::setFitness(int newFit)
{
	m_fitness = newFit;
}

int GeneticalLearning::getFitness()
{
	return m_fitness;
}

float GeneticalLearning::randomWeight()
{
	switch (m_FActivation)
	{
	case SIGMOIDE:
		return RandomFloatInRange(0.0f, 1.0f);

	case TANGEANTE_HYPER:
		return RandomFloatInRange(-1.0f, 1.0f);

	default:
		return RandomFloatInRange(0.0f, 1.0f);
	}
}

float GeneticalLearning::RandomFloatInRange(float min, float max)
{
	return (min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min))));
}
