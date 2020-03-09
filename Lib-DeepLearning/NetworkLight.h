#pragma once

#include<fstream>
#include<cmath>
#include<ctype.h>

template<const int nbPerceptrons, const int nbWeights>
class NetworkLight
{
private:
	float m_weights[nbWeights];
	float m_biais[nbPerceptrons];
	float m_perceptrons[nbPerceptrons];
	float *m_input;
	const uint8_t *m_sizeHidden;
	const uint8_t m_nbInput;
	const uint8_t m_nbOutput;
	const uint8_t m_nbHidden;
	bool m_deleteResult;

	uint8_t getSize(uint8_t i);
	int getIndexDouble(uint8_t i, uint8_t j);
	int getIndexTriple(uint8_t i, uint8_t j, uint8_t k);

public:
	void setDeleteParam(bool deleteResult);
	float* runPrediction(float *input);
	bool loadWeight(const char* path);
	NetworkLight(uint8_t* sizeHidden, uint8_t nbInput, uint8_t nbOutput, uint8_t nbHidden);

	static float F_Sigmoide(float value);
};

template<int nbPerceptrons, int nbWeights>
inline uint8_t NetworkLight<nbPerceptrons, nbWeights>::getSize(uint8_t i)
{
	if (i == 0)
		return m_sizeHidden[0];

	return (m_sizeHidden[i] - m_sizeHidden[i - 1]);
}

template<int nbPerceptrons, int nbWeights>
inline int NetworkLight<nbPerceptrons, nbWeights>::getIndexDouble(uint8_t i, uint8_t j)
{
	if (i == 0)
		return j;

	return m_sizeHidden[i - 1] + j;
}

template<int nbPerceptrons, int nbWeights>
inline int NetworkLight<nbPerceptrons, nbWeights>::getIndexTriple(uint8_t i, uint8_t j, uint8_t k)
{
	return 0;
}

template<int nbPerceptrons, int nbWeights>
inline void NetworkLight<nbPerceptrons, nbWeights>::setDeleteParam(bool deleteResult)
{
	m_deleteResult = deleteResult;
}

template<int nbPerceptrons, int nbWeights>
inline float* NetworkLight<nbPerceptrons, nbWeights>::runPrediction(float* input)
{
	m_input = input;
	float somme = 0.0f;
	for (int i = 0; i < getSize(0); i++)
	{
		somme = 0;
		for (int j = 0; j < m_nbInput; j++)
			somme += m_weights[getIndexTriple(0, i, j)] * m_input[j];
		somme += m_biais[getIndexDouble(0, i)];
		m_perceptrons[getIndexDouble(0, i)] = F_Sigmoide(somme);
	}

	for (int a = 1; a < m_nbHidden; a++)
	{
		for (int i = 0; i < getSize(a); i++)
		{
			somme = 0.0f;
			for (int j = 0; j < getSize(a - 1); j++)
				somme += m_weights[getIndexTriple(a, i, j)] * m_perceptrons[getIndexDouble(a - 1, j)];
			somme += m_biais[getIndexDouble(a, i)];
			m_perceptrons[getIndexDouble(a, i)] = F_Sigmoide(somme);
		}
	}

	for (int w = 0; w < getSize(m_nbHidden); w++)
	{
		somme = 0.0f;
		for (int j = 0; j < getSize(m_nbHidden - 1); j++)
			somme += m_weights[getIndexTriple(m_nbHidden, w, j)] * m_perceptrons[getIndexDouble(m_nbHidden - 1, j)];
		somme += m_biais[getIndexDouble(m_nbHidden, w)];
		m_perceptrons[getIndexDouble(m_nbHidden, w)] = somme;
	}

	return (m_perceptrons + getIndexDouble(m_nbHidden, 0));
}

template<int nbPerceptrons, int nbWeights>
inline bool NetworkLight<nbPerceptrons, nbWeights>::loadWeight(const char* path)
{
	std::ifstream file;
	file.open(path);
	for(int i=0;i<m_nbHidden;i++)
		for(int j=0; j<getSize(i);j++)

	if (!file)
		return false;
}

template<int nbPerceptrons, int nbWeights>
inline NetworkLight<nbPerceptrons, nbWeights>::NetworkLight(uint8_t*sizeHidden, uint8_t nbInput, uint8_t nbOutput, uint8_t nbHidden):m_sizeHidden(sizeHidden), m_nbInput(nbInput), m_nbOutput(nbOutput), m_nbHidden(nbHidden), m_deleteResult(true)
{}

template<int nbPerceptrons, int nbWeights>
inline float NetworkLight<nbPerceptrons, nbWeights>::F_Sigmoide(float value)
{
	float inverse = pow(2.718281828459, value);
	inverse = 1 + (1 / inverse);
	return 1 / inverse;
}
