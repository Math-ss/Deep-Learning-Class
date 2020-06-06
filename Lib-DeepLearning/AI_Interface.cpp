/*
"AI_Interface.cpp"

Created by Math's
*/

#include "AI_Interface.h"

void AI_Interface::training(TrainingParameters *param)
{
	for (int i = 0; i < param->repetition; i++)
	{
		computePrediction(param, &((*m_data)[i % m_data->size()]));
		updateParameters(param);
	}
}

void AI_Interface::setData(std::vector<std::vector<float>> *data)
{
	m_data = data;
}

void AI_Interface::setExcepted(std::vector<std::vector<float>> *excepted)
{
	m_excepted = excepted;
}
