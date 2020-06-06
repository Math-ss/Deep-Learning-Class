/*
"AI_Interface.h"

Created by Math's
*/

#pragma once
#include <vector>
/*
Basic interface to implement an AI
*/
class AI_Interface
{
public:
	AI_Interface();

	virtual std::vector<float>& getPredictionRef() const = 0;
	virtual std::vector<float> getPredictionCopy() const = 0;

	/*
	Just calculate the results and store them within the class.
	*/
	virtual void computePrediction(std::vector<float> &input) = 0;
	virtual void updateParameters(float learningRate = 0.5) = 0;
	/*
	Called after computePrediction() during training (unless training is overridden)
	Used to improve AI (change wheights,...)
	*/

	/*
	Basically, just call computePrediction() then updateParameters() param->repetition times
	*/
	virtual void training(int repetition = 1, float learningRate = 0.5);

	/*
	Set m_data
	*/
	virtual void setData(std::vector<std::vector<float>>& data);
	virtual void setExcepted(std::vector<std::vector<float>>& excepted);
	/*
	Set m_excepted
	*/

protected:
	std::vector<std::vector<float>>& m_data;
	std::vector<std::vector<float>>& m_excepted;
	/*
	Trainig data
	*/
	/*
	Expected results for these data
	*/
};

