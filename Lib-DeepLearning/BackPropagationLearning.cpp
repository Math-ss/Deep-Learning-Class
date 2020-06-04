/*
deepBackPropagationLearning.cpp

Be careful, the softmax mode isn't good !!!!

Create by 
MATHGIQUE
*/

#include<cstdio>
#include<vector>
#include<string>
#include<fstream>
#include <iostream>
#include "BackPropagationLearning.h"


using namespace std;

void BackPropagationLearning::training(int repetition, float learningRate)
{
	for (int i = 0; i < repetition; i++)
	{
		m_trainingProgression = i % m_data.size();
		computePrediction(m_data[i % m_data.size()]);
		updateParameters(learningRate);
	}
}

void BackPropagationLearning::updateParameters(float learningRate)
{
	int last = m_perceptrons.size() - 1;
	
	if (m_fLogi == "tangente" || m_fLogi == "sigmoide" || m_fLogi == "softmax")
	{
		for (int i = 0; i < m_perceptrons[last].size(); i++)
		{
			m_delta[last][i] = (m_excepted[m_trainingProgression][i] - m_perceptrons[last][i]) * F_Derivative_Sigmoide(m_perceptrons[last][i]);
			for (int j = 0; j < m_perceptrons[last - 1].size(); j++)
				m_weights[last][i][j] += learningRate * m_delta[last][i] * m_perceptrons[last - 1][j];
			m_biais[last][i] += learningRate * m_delta[last][i];

			//printf("%f\n", m_perceptrons[last][i]);
		}
		//printf("\n");
	}
	
	for (int i = last - 1; i > 0; i--)
	{
		for (int j = 0; j < m_perceptrons[i].size(); j++)
		{
			float error = 0;
			for (int k = 0; k < m_perceptrons[i + 1].size(); k++)
				error += m_weights[i + 1][k][j] * m_delta[i + 1][k];

			m_delta[i][j] = error * F_Derivative_Sigmoide(m_perceptrons[i][j]);

			for (int k = 0; k < m_perceptrons[i - 1].size(); k++)
				m_weights[i][j][k] += learningRate * m_delta[i][j] * m_perceptrons[i - 1][k];
			m_biais[i][j] += learningRate * m_delta[i][j];
		}
	}

	for (int j = 0; j < m_perceptrons[0].size(); j++)
	{
		float error = 0;
		for (int k = 0; k < m_perceptrons[1].size(); k++)
			error += m_weights[1][k][j] * m_delta[1][k];

		m_delta[0][j] = error * F_Derivative_Sigmoide(m_perceptrons[0][j]);

		for (int k = 0; k < m_data[m_trainingProgression].size(); k++)
			m_weights[0][j][k] += 0.5 * m_delta[0][j] * m_data[m_trainingProgression][k];
		m_biais[0][j] += 0.5 * m_delta[0][j];
	}
}

BackPropagationLearning::BackPropagationLearning(int inputLayer, int outputLayer, vector<int>& hiddenLayer, string& f)
	:DeepLearningNetwork::DeepLearningNetwork(inputLayer, outputLayer, hiddenLayer, f), m_trainingProgression(0)
{
	m_delta.push_back(vector<float>(hiddenLayer[0]));

	for (int i = 1; i < hiddenLayer.size(); i++)
		m_delta.push_back(vector<float>(hiddenLayer[i]));

	m_delta.push_back(vector<float>(outputLayer));
}