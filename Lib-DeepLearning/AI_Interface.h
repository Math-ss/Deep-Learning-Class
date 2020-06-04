#include <vector>

#pragma once
class AI_Interface
{
public:
	AI_Interface();

	virtual std::vector<float>& getPredictionRef() const = 0;
	virtual std::vector<float> getPredictionCopy() const = 0;

	virtual void computePrediction(std::vector<float> &input) = 0;
	virtual void updateParameters(float learningRate = 0.5) = 0;

	virtual void training(int repetition = 1, float learningRate = 0.5);

	virtual void setData(std::vector<std::vector<float>>& data);
	virtual void setExcepted(std::vector<std::vector<float>>& excepted);

protected:
	std::vector<std::vector<float>>& m_data;
	std::vector<std::vector<float>>& m_excepted;
};

