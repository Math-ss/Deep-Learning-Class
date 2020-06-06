/*
"AI_Interface.cpp"

Created by Math's
*/

#include "AI_Interface.h"

void AI_Interface::training(int repetition, float learningRate = 0.5)
{
	for (int i = 0; i < repetition; i++)
	{
		computePrediction(m_data[i % m_data.size()]);
		updateParameters(learningRate);
	}
}

void AI_Interface::setData(std::vector<std::vector<float>>& data)
{
	m_data = data;
}

void AI_Interface::setExcepted(std::vector<std::vector<float>>& excepted)
{
	m_excepted = excepted;
}
