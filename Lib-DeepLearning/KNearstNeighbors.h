/*
"KNearstNeighbors.h"

Created by Math's
*/

#pragma once

#include<map>
#include "AI_Interface.h"

typedef class KNearstNeighbors K_NN;

/*
Simple typedef that uses the fact that maps are ordered according to keys to find the k closest neighbors more easily.
It is a multimap in case points are equidistant from the studied point.
*/
typedef std::multimap<float, int> ordered_index;

struct K_NNParameters : public TrainingParameters
{
	/*
	Indicates the k value to use if greater than 0
	*/
	int k;

	K_NNParameters() : k(1) {}
};

class KNearstNeighbors : public AI_Interface
{
protected:
	std::vector<float> m_results;

	/*
	Basic k, can be override by param->k.
	*/
	int m_k;

	/*
	Current minimal error obtained during a training process
	*/
	float m_minimalError;

	/*
	current error during a training process (reset at the end)
	*/
	float m_currentError;

	/*
	Training progression for a k value
	*/
	int m_trainingProgression;

public:
	KNearstNeighbors();
	KNearstNeighbors(KNearstNeighbors& source);

	/*
	Return a pointer of m_results
	*/
	virtual std::vector<float> const* getPredictionPtr() const override;

	/*
	Just return m_results
	*/
	virtual std::vector<float> getPrediction() const override;

	/*
	Make K_NN algorithm
		*param must be of type K_NNParameters
			-uses param->k if different to 0, else the function uses m_k (by default 1)
	*/
	virtual void computePrediction(TrainingParameters* param, std::vector<float>* input) override;

	/*
	Modifies the current error in accordance with the latest results and modifies k if necessary (at the end of the training by a value of k)
		*param must be of type K_NNParameters
	*/
	virtual void updateParameters(TrainingParameters* param) override;

	/*
	Try to find the better k :
	Just increment param->k param->repeat times.
	Call each time computeResults() then updateParameters() for each point of m_data.
		*param must be of type K_NNParameters
			-starts at param->k if it is greater than 0 (else starts at m_k)
			-param->repetition indicates the number of k values tested
	*/
	virtual void training(TrainingParameters* param) override;
};

