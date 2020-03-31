/*
deepNetworkMini.cpp

Be careful, the softmax mode isn't good !!!!

Create by 
MATHGIQUE
*/

#include<cstdio>
#include<vector>
#include<string>
#include<fstream>
#include "deepNetworkMini.h"


using namespace std;

void NetworkMini::feedForward()
{
	double somme = 0;
	int last = m_perceptrons.size() - 1;

	if (last > 0)
	{
		for (int i = 0; i < m_perceptrons[0].size(); i++)
		{
			somme = 0;
			for (int j = 0; j < m_input->size(); j++)
				somme += m_weights[0][i][j] * (*m_input)[j];
			somme += m_biais[0][i];
			m_perceptrons[0][i] = F_Sigmoide(somme);
		}

		for (int a = 1; a < last; a++)
		{
			for (int i = 0; i < m_perceptrons[a].size(); i++)
			{
				somme = 0;
				for (int j = 0; j < m_perceptrons[a - 1].size(); j++)
					somme += m_weights[a][i][j] * m_perceptrons[a - 1][j];
				somme += m_biais[a][i];
				m_perceptrons[a][i] = F_Sigmoide(somme);
			}
		}

		for (int w = 0; w < m_perceptrons[last].size(); w++)
		{
			somme = 0;
			for (int j = 0; j < m_perceptrons[last - 1].size(); j++)
				somme += m_weights[last][w][j] * m_perceptrons[last - 1][j];
			somme += m_biais[last][w];
			m_perceptrons[last][w] = somme;
		}
	}
	else
	{
		for (int w = 0; w < m_perceptrons[0].size(); w++)
		{
			somme = 0;
			for (int j = 0; j < m_input->size(); j++)
				somme += m_weights[0][w][j] * (*m_input)[j];
			somme += m_biais[0][w];
			m_perceptrons[0][w] = somme;
		}
	}
	
	if (m_fLogi == "sigmoide")
	{
		for (int w = 0; w < m_perceptrons[last].size(); w++)
			m_perceptrons[last][w] = F_Sigmoide(m_perceptrons[last][w]);
	}
	else if (m_fLogi == "softmax")
	{
		F_Softmax(&(m_perceptrons[last]));
	}
	else if (m_fLogi == "tangente")
	{
		for (int w = 0; w < m_perceptrons[0].size(); w++)
			m_perceptrons[last][w] = F_Tangente(m_perceptrons[last][w]);
	}

	return;
}

void NetworkMini::backPropagation()
{
	int last = m_perceptrons.size() - 1;
	
	if (m_fLogi == "tangente" || m_fLogi == "sigmoide" || m_fLogi == "softmax")
	{
		for (int i = 0; i < m_perceptrons[last].size(); i++)
		{
			m_delta[last][i] = ((*m_excepted)[i] - m_perceptrons[last][i]) * F_Prime(m_perceptrons[last][i]);
			for (int j = 0; j < m_perceptrons[last - 1].size(); j++)
				m_weights[last][i][j] += m_coefficient * m_delta[last][i] * m_perceptrons[last - 1][j];
			m_biais[last][i] += m_coefficient * m_delta[last][i];

			if (m_perceptrons[last][i] > 0.5)
				m_nb++;

			//printf("%f\n", m_perceptrons[last][i]);
		}
		//printf("\n");
	}
	
	for (int i = last - 1; i > 0; i--)
	{
		for (int j = 0; j < m_perceptrons[i].size(); j++)
		{
			double error = 0;
			for (int k = 0; k < m_perceptrons[i + 1].size(); k++)
				error += m_weights[i + 1][k][j] * m_delta[i + 1][k];

			m_delta[i][j] = error * F_Prime(m_perceptrons[i][j]);

			for (int k = 0; k < m_perceptrons[i - 1].size(); k++)
				m_weights[i][j][k] += m_coefficient * m_delta[i][j] * m_perceptrons[i - 1][k];
			m_biais[i][j] += m_coefficient * m_delta[i][j];
		}
	}

	for (int j = 0; j < m_perceptrons[0].size(); j++)
	{
		double error = 0;
		for (int k = 0; k < m_perceptrons[1].size(); k++)
			error += m_weights[1][k][j] * m_delta[1][k];

		m_delta[0][j] = error * F_Prime(m_perceptrons[0][j]);

		for (int k = 0; k < m_input->size(); k++)
			m_weights[0][j][k] += 0.5 * m_delta[0][j] * (*m_input)[k];
		m_biais[0][j] += 0.5 * m_delta[0][j];
	}
}

void NetworkMini::setInput(std::vector<double>* input)
{
  m_input = input;
}

void NetworkMini::setExcepted(std::vector<double>* excepted)
{
	m_excepted = excepted;
}

NetworkMini::NetworkMini(int inputLayer, int outputLayer, vector<int> &hiddenLayer, string &f):m_nbInput(inputLayer), m_nbOutput(outputLayer), m_fLogi(f)
{
	m_nb = 0;
	m_coefficient = 0.5;
	
	m_perceptrons.push_back(vector<double>(hiddenLayer[0]));
	m_biais.push_back(vector<double>(hiddenLayer[0]));
	m_delta.push_back(vector<double>(hiddenLayer[0]));
	m_weights.push_back(vector<vector<double>>(hiddenLayer[0], vector<double>(m_nbInput)));

	for (int i = 1; i < hiddenLayer.size(); i++)
	{
		m_perceptrons.push_back(vector<double>(hiddenLayer[i]));
		m_biais.push_back(vector<double>(hiddenLayer[i]));
		m_delta.push_back(vector<double>(hiddenLayer[i]));
		m_weights.push_back(vector<vector<double>>(hiddenLayer[i], vector<double>(hiddenLayer[i - 1])));
	}

	m_perceptrons.push_back(vector<double>(outputLayer));
	m_biais.push_back(vector<double>(outputLayer));
	m_delta.push_back(vector<double>(outputLayer));
	m_weights.push_back(vector<vector<double>>(outputLayer, vector<double>(hiddenLayer[hiddenLayer.size() - 1])));

	for (int x = 0; x < m_weights.size(); x++)
		for (int y = 0; y < m_weights[x].size(); y++)
		{
			for (int z = 0; z < m_weights[x][y].size(); z++)
				m_weights[x][y][z] = rand() / 32767.0000;
			m_biais[x][y] = rand() / 32767.0000;
		}

}

void NetworkMini::runLearning(vector<vector<double>>* test, vector<std::vector<double>>* resultDesired)
{
	int nb_test = test->size();
	m_nb = 0;
	for (int g = 0; g < nb_test; g++)
	{
		setExcepted(&((*resultDesired)[g]));
		setInput(&((*test)[g]));
		
		feedForward();
		backPropagation();
	}
	printf("%d\n", m_nb);
	//printf("End learning\n");
}

void NetworkMini::runPrediction(std::vector<double>* data)
{
	setInput(data);
	feedForward();
	for (int h = 0; h < m_perceptrons[m_perceptrons.size() - 1].size(); h++)
		printf("%f\n", m_perceptrons[m_perceptrons.size() - 1][h]);
	printf("\n");
}

bool NetworkMini::setCoefficient(double value)
{
  if (value < 1 && value > 0)
  {
	  m_coefficient = value;
	  return true;
  }
  else
	return false;
}

double NetworkMini::F_Sigmoide(double value)
{
	double inverse = pow(2.718281828459, value);
	inverse = 1 + (1 / inverse);
	return 1 / inverse;
}

double NetworkMini::F_Prime(double value)
{
	return value * (1.0 - value);
}

double NetworkMini::F_Tangente(double value)
{
	double num, denom, inverse;
	inverse = pow(2.718281828459, 2 * value);
	inverse = 1 / inverse;
	num = 1 - inverse;
	denom = 1 + inverse;
	return (num / denom);
}

void NetworkMini::F_Softmax(vector<double>* value)
{
	double denom = 0.0000, num = 0.00;
	for (int i = 0; i < value->size(); i++)
		denom += pow(2.718281828459, (*value)[i]);

	for (int i = 0; i < value->size(); i++)
	{
		num = pow(2.718281828459, (*value)[i]);
		(*value)[i] = num / denom;
	}
}

bool NetworkMini::saveWeights(string path)
{
	path += "info.wtw";

	ofstream file;
	file.open("path", std::ofstream::out | std::ofstream::trunc);

	if (!file)
		return false;

	for (int i = 0; i < m_weights.size(); i++)
		for (int j = 0; j < m_weights[i].size(); j++)
			for (int k = 0; k < m_weights[i][j].size(); k++)
				file << m_weights[i][j][k] << std::endl;

	for (int i = 0; i < m_weights.size(); i++)
		for (int j = 0; j < m_weights[i].size(); j++)
			file << m_biais[i][j] << std::endl;

	return true;
}