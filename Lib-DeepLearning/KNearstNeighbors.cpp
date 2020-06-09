/*
"KNearstNeighbors.cpp"

Created by Math's
*/

#include <limits>
#include "KNearstNeighbors.h"

using namespace std;

KNearstNeighbors::KNearstNeighbors() : m_results(1), m_k(1), m_minimalError(numeric_limits<float>::max()), m_currentError(0.0f), m_trainingProgression(-1)
{}

KNearstNeighbors::KNearstNeighbors(KNearstNeighbors & source) : m_results(source.m_results), m_k(source.m_k), m_minimalError(source.m_minimalError), m_currentError(0.0f), m_trainingProgression(-1)
{}

std::vector<float> const* KNearstNeighbors::getPredictionPtr() const
{
	return &(m_results);
}

std::vector<float> KNearstNeighbors::getPrediction() const
{
	return m_results;
}

void KNearstNeighbors::computePrediction(TrainingParameters* param, std::vector<float>* input)
{
	K_NNParameters* K_NNParam = static_cast<K_NNParameters*> (param);
	int k = (K_NNParam->k <= 0) ? m_k : K_NNParam->k;

	ordered_index NearstNeighbors;//Use multimap order to find the K NN
	float somme = 0.0f;

	for (int i = 0; i < m_data->size(); i++)//Calculation of all distances
	{
		somme = 0.0f;
		for (int j = 0; j < (*m_data)[i].size(); j++)
		{
			somme += powf( ((*m_data)[i][j] - (*input)[j]) , 2.0f); //Euclidean distance in (*m_data)[i].size() dimensions
		}

		if (somme == 0 && m_trainingProgression == i) //In case of training check that we do not compare the same points.
			continue;

		somme = sqrtf(somme);
		NearstNeighbors.insert(pair<float, int>(somme, i)); 
	}

	int i = 0;
	somme = 0.0f;

	for (ordered_index::iterator it = NearstNeighbors.begin(); it != NearstNeighbors.end(); it++)//Find the smallest k distances by using iteraror
	{
		if (i == k)
			break;

		somme += (*m_excepted)[it->second][0];
	}

	m_results[0] = somme / k; //Regression result
}

void KNearstNeighbors::updateParameters(TrainingParameters* param)
{
	K_NNParameters* K_NNParam = static_cast<K_NNParameters*> (param);
	
	if (m_trainingProgression == m_data->size()) //If training session for a k value is over
	{
		m_currentError /= m_data->size();

		if (m_currentError < m_minimalError)
		{
			m_minimalError = m_currentError;
			m_k = K_NNParam->k;
		}

		m_currentError = 0.0f;
		m_trainingProgression = 0;
	}
	else
	{
		m_currentError += (m_results[0] - (*m_excepted)[m_trainingProgression][0]) * (m_results[0] - (*m_excepted)[m_trainingProgression][0]);
	}
}

void KNearstNeighbors::training(TrainingParameters* param)
{
	K_NNParameters* K_NNParam = static_cast<K_NNParameters*> (param);
	
	if (K_NNParam->k <= 0)
		K_NNParam->k = m_k;
	
	for (int i = 0; i < param->repetition; i++)
	{
		if (K_NNParam->k >= m_data->size())
			break;
		
		for (m_trainingProgression = 0; m_trainingProgression < m_data->size(); m_trainingProgression++)
		{
			computePrediction(param, &((*m_data)[m_trainingProgression]));
			updateParameters(param);
		}

		updateParameters(param);
		K_NNParam->k++;
	}

	m_currentError = 0.0f;
	m_trainingProgression = -1;
}
