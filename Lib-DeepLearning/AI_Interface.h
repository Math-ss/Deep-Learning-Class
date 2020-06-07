/*
"AI_Interface.h"

Created by Math's
*/

#pragma once

#include <vector>

/*
Contains all the arguments that the user has to give to the 
training/computePrediction/updateParameters function.
*/

struct TrainingParameters
{
	/*
	Number of times the training will be repeated
	*/
	int repetition;

	TrainingParameters() : repetition(1) {}
};

/*
Basic interface to implement an AI
*/
class AI_Interface
{
public:
	AI_Interface() : m_data(nullptr), m_excepted(nullptr) {}
	AI_Interface(AI_Interface& source) : m_data(nullptr), m_excepted(nullptr) {}

	/*
	Just return a pointer of the latest results obtained 
	*/
	virtual std::vector<float> const *getPredictionPtr() const = 0;

	/*
	Just return a copy (or a const ref) of the latest results obtained
	*/
	virtual std::vector<float> getPrediction() const = 0;

	/*
	Just calculate the results and store them within the class.
	*/
	virtual void computePrediction(TrainingParameters *param, std::vector<float> *input) = 0;
	
	/*
	Called after computePrediction() during training (unless training is overridden)
	Used to improve AI (change wheights,...)
	*/
	virtual void updateParameters(TrainingParameters *param) = 0;

	/*
	Basically, just call computePrediction() then updateParameters() param->repetition times
	*/
	virtual void training(TrainingParameters *param);

	/*
	Set m_data
	*/
	virtual void setData(std::vector<std::vector<float>> *data);

	/*
	Set m_excepted
	*/
	virtual void setExcepted(std::vector<std::vector<float>> *excepted);

protected:

	/*
	Trainig data
	*/
	std::vector<std::vector<float>> *m_data;

	/*
	Expected results for these data
	*/
	std::vector<std::vector<float>> *m_excepted;
};

