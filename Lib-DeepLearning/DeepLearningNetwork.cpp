/*
"DeepLearningNetwork.cpp"

Created by Math's
*/

#include <fstream>
#include <iostream>
#include "DeepLearningNetwork.h"

using namespace std;

DeepLearningNetwork::DeepLearningNetwork(int inputLayer, int outputLayer, std::vector<int>& hiddenLayer, AIF_Activation FActivation)
	:m_nbInput(inputLayer), m_nbOutput(outputLayer), m_FActivation(FActivation)
{
	m_perceptrons.push_back(vector<float>(hiddenLayer[0]));
	m_biais.push_back(vector<float>(hiddenLayer[0]));
	m_weights.push_back(vector<vector<float>>(hiddenLayer[0], vector<float>(m_nbInput)));

	for (int i = 1; i < hiddenLayer.size(); i++)
	{
		m_perceptrons.push_back(vector<float>(hiddenLayer[i]));
		m_biais.push_back(vector<float>(hiddenLayer[i]));
		m_weights.push_back(vector<vector<float>>(hiddenLayer[i], vector<float>(hiddenLayer[i - 1])));
	}

	m_perceptrons.push_back(vector<float>(outputLayer));
	m_biais.push_back(vector<float>(outputLayer));
	m_weights.push_back(vector<vector<float>>(outputLayer, vector<float>(hiddenLayer[hiddenLayer.size() - 1])));

	for (int x = 0; x < m_weights.size(); x++)
		for (int y = 0; y < m_weights[x].size(); y++)
		{
			for (int z = 0; z < m_weights[x][y].size(); z++)
				m_weights[x][y][z] = rand() / ((float) RAND_MAX);
			m_biais[x][y] = rand() / ((float)RAND_MAX);
		}
}

DeepLearningNetwork::DeepLearningNetwork(DeepLearningNetwork& source)
	:AI_Interface(source), m_weights(source.m_weights), m_biais(source.m_biais), m_perceptrons(source.m_perceptrons), m_nbInput(source.m_nbInput), m_nbOutput(source.m_nbOutput), m_FActivation(source.m_FActivation)
{}

void DeepLearningNetwork::computePrediction(TrainingParameters *param, std::vector<float> *input)
{
	float somme = 0;
	int last = m_perceptrons.size() - 1;

	if (last > 0)
	{
		for (int i = 0; i < m_perceptrons[0].size(); i++)
		{
			somme = 0.0f;
			for (int j = 0; j < input->size(); j++)
				somme += m_weights[0][i][j] * (*input)[j];
			somme += m_biais[0][i];
			m_perceptrons[0][i] = F_Activation(somme);
		}

		for (int a = 1; a < last; a++)
		{
			for (int i = 0; i < m_perceptrons[a].size(); i++)
			{
				somme = 0.0f;
				for (int j = 0; j < m_perceptrons[a - 1].size(); j++)
					somme += m_weights[a][i][j] * m_perceptrons[a - 1][j];
				somme += m_biais[a][i];
				m_perceptrons[a][i] = F_Activation(somme);
			}
		}

		for (int w = 0; w < m_perceptrons[last].size(); w++)
		{
			somme = 0.0f;
			for (int j = 0; j < m_perceptrons[last - 1].size(); j++)
				somme += m_weights[last][w][j] * m_perceptrons[last - 1][j];
			somme += m_biais[last][w];
			m_perceptrons[last][w] = F_Activation(somme);
		}
	}
	else
	{
		for (int w = 0; w < m_perceptrons[0].size(); w++)
		{
			somme = 0.0f;
			for (int j = 0; j < input->size(); j++)
				somme += m_weights[0][w][j] * (*input)[j];
			somme += m_biais[0][w];
			m_perceptrons[0][w] = F_Activation(somme);
		}
	}
}

std::vector<float> const *DeepLearningNetwork::getPredictionPtr() const
{
	return &(m_perceptrons[m_perceptrons.size() - 1]);
}

std::vector<float> DeepLearningNetwork::getPrediction() const
{
	return m_perceptrons[m_perceptrons.size() - 1];
}

float DeepLearningNetwork::F_Sigmoide(float value)
{
	float inverse = powf(2.718281828459f, value);
	inverse = 1.0f + (1.0f / inverse);
	return 1.0f / inverse;
}

float DeepLearningNetwork::F_TangenteHyper(float value)
{
	float num, denom, inverse;
	inverse = powf(2.718281828459f, 2.0f * value);
	inverse = 1.0f / inverse;
	num = 1.0f - inverse;
	denom = 1.0f + inverse;
	return (num / denom);
}

float DeepLearningNetwork::F_Activation(float value)
{
	switch (m_FActivation)
	{
	case SIGMOIDE:
		return F_Sigmoide(value);

	case TANGEANTE_HYPER:
		return F_TangenteHyper(value);

	default:
		return F_Sigmoide(value);
	}
}

/*void DeepLearningNetwork::F_Softmax(std::vector<float>& value)
{
	float denom = 0.0f, num = 0.0f;

	for (int i = 0; i < value.size(); i++)
		denom += pow(2.718281828459f, value[i]);

	for (int i = 0; i < value.size(); i++)
	{
		num = pow(2.718281828459f, value[i]);
		value[i] = num / denom;
	}
}*/

bool DeepLearningNetwork::saveWeights(std::string path)
{
	path += "info.wtw";

	ofstream file;
	file.open("path", std::ofstream::out | std::ofstream::trunc);

	if (!file)
	{
		std::cerr << "Can't open the save file" << std::endl;
		return false;
	}

	for (int i = 0; i < m_weights.size(); i++)
		for (int j = 0; j < m_weights[i].size(); j++)
			for (int k = 0; k < m_weights[i][j].size(); k++)
				file << m_weights[i][j][k] << std::endl;

	for (int i = 0; i < m_weights.size(); i++)
		for (int j = 0; j < m_weights[i].size(); j++)
			file << m_biais[i][j] << std::endl;

	file.close();

	return true;
}
