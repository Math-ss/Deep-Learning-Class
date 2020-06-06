/*
"BackPropagationLearning.cpp"

Created by Math's
*/

#include<cstdio>
#include<vector>
#include<string>
#include<fstream>
#include <iostream>
#include "BackPropagationLearning.h"

using namespace std;

void BackPropagationLearning::training(TrainingParameters* param)
{
	for (int i = 0; i < param->repetition; i++)
	{
		m_trainingProgression = i % m_data->size();
		computePrediction(param, &((*m_data)[m_trainingProgression]));
		updateParameters(param);
	}
}

void BackPropagationLearning::updateParameters(TrainingParameters* param)
{
	int last = m_perceptrons.size() - 1;
	BackPropagationParameters* BackParam = (BackPropagationParameters*) param;

	for (int i = 0; i < m_perceptrons[last].size(); i++)
	{
		m_delta[last][i] = ((*m_excepted)[m_trainingProgression][i] - m_perceptrons[last][i]) * F_Derivative_Activation(m_perceptrons[last][i]);
		for (int j = 0; j < m_perceptrons[last - 1].size(); j++)
			m_weights[last][i][j] += BackParam->learningRate * m_delta[last][i] * m_perceptrons[last - 1][j];
		m_biais[last][i] += BackParam->learningRate * m_delta[last][i];

		//printf("%f\n", m_perceptrons[last][i]);
	}
	//printf("\n");

	for (int i = last - 1; i > 0; i--)
	{
		for (int j = 0; j < m_perceptrons[i].size(); j++)
		{
			float error = 0;
			for (int k = 0; k < m_perceptrons[i+1].size(); k++)
				error += m_weights[i+1][k][j] * m_delta[i+1][k];

			m_delta[i][j] = error * F_Derivative_Activation(m_perceptrons[i][j]);

			for (int k = 0; k < m_perceptrons[i-1].size(); k++)
				m_weights[i][j][k] += BackParam->learningRate * m_delta[i][j] * m_perceptrons[i - 1][k];
			m_biais[i][j] += BackParam->learningRate * m_delta[i][j];
		}
	}

	for (int j = 0; j < m_perceptrons[0].size(); j++)
	{
		float error = 0;
		for (int k = 0; k < m_perceptrons[1].size(); k++)
			error += m_weights[1][k][j] * m_delta[1][k];

		m_delta[0][j] = error * F_Derivative_Activation(m_perceptrons[0][j]);

		for (int k = 0; k < (*m_data)[m_trainingProgression].size(); k++)
			m_weights[0][j][k] += BackParam->learningRate * m_delta[0][j] * (*m_data)[m_trainingProgression][k];
		m_biais[0][j] += BackParam->learningRate * m_delta[0][j];
	}
}

BackPropagationLearning::BackPropagationLearning(int inputLayer, int outputLayer, vector<int>& hiddenLayer, AIF_Activation FActivation)
	:DeepLearningNetwork::DeepLearningNetwork(inputLayer, outputLayer, hiddenLayer, FActivation), m_trainingProgression(0)
{
	m_delta.push_back(vector<float>(hiddenLayer[0]));

	for (int i = 1; i < hiddenLayer.size(); i++)
		m_delta.push_back(vector<float>(hiddenLayer[i]));

	m_delta.push_back(vector<float>(outputLayer));
}

float BackPropagationLearning::F_Derivative_Sigmoide(float value)
{
	return value * (1.0f - value);
}

float BackPropagationLearning::F_Derivative_TangenteHyper(float value)
{
	return 1.0f - powf(value, 2.0f);
}

float BackPropagationLearning::F_Derivative_Activation(float value)
{
	switch (m_FActivation)
	{
	case SIGMOIDE:
		return F_Derivative_Sigmoide(value);

	case TANGEANTE_HYPER:
		return F_Derivative_TangenteHyper(value);
	default:
		return F_Derivative_Sigmoide(value);
	}
}
