/*
deepBackPropagationLearning.h

int m_nb is for debuging

Created by
MATHGIQUE
*/

#pragma once

#include<fstream>
#include<cmath>
#include<ctype.h>

template<typename indexType, const int nbPerceptrons, const int nbWeights>
class NetworkLight
{
private:
	float m_weights[nbWeights];
	float m_biais[nbPerceptrons];
	float m_perceptrons[nbPerceptrons];

protected:
	float* m_input;
	const indexType *m_sizeHidden;//Permet de gerer les index
	const indexType m_nbInput;//Permet de gerer les index
	const indexType m_nbOutput;//Permet de gerer les index
	const indexType m_nbHidden;//Permet de gerer les index

	indexType getSize(indexType i);
	int getIndexDouble(indexType i, indexType j);
	int getIndexTriple(indexType i, indexType j, indexType k);

public:
	virtual float* runPrediction(float *input) final;
	virtual bool loadWeight(const char* path);
	NetworkLight(indexType* sizeHidden, indexType nbInput, indexType nbOutput, indexType nbHidden);

	static float F_Sigmoide(float value);
};

template<typename indexType, int nbPerceptrons, int nbWeights>
inline indexType NetworkLight<indexType, nbPerceptrons, nbWeights>::getSize(indexType i)
{
	if (i == 0)
		return m_sizeHidden[0];

	return (m_sizeHidden[i] - m_sizeHidden[i - 1]);
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline int NetworkLight<indexType, nbPerceptrons, nbWeights>::getIndexDouble(indexType i, indexType j)
{
	if (i == 0)
		return j;

	return m_sizeHidden[i - 1] + j;
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline int NetworkLight<indexType, nbPerceptrons, nbWeights>::getIndexTriple(indexType i, indexType j, indexType k)
{
	if (i == 0)
		return ((j * m_nbInput) + k);

	int before = getSize(0) * m_nbInput;
	for (int x = 1; x < i; x++)
		before += getSize(x) * getSize(x - 1);

	return (before + (j * getSize(i-1)) + k);
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline float* NetworkLight<indexType, nbPerceptrons, nbWeights>::runPrediction(float* input)
{
	m_input = input;
	float somme = 0.0f;
	float bufferTest = 0.0f;

	for (int i = 0; i < getSize(0); i++)
	{
		somme = 0.0f;
		for (int j = 0; j < m_nbInput; j++)
		{
			bufferTest = m_input[j];
			somme += m_weights[getIndexTriple(0, i, j)] * bufferTest;
		}
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

	for (int i = 0; i < m_nbOutput; i++)
	{
		somme = 0.0f;
		for (int j = 0; j < getSize(m_nbHidden - 1); j++)
			somme += m_weights[getIndexTriple(m_nbHidden, i, j)] * m_perceptrons[getIndexDouble(m_nbHidden - 1, j)];
		somme += m_biais[getIndexDouble(m_nbHidden, i)];
		m_perceptrons[getIndexDouble(m_nbHidden, i)] = F_Sigmoide(somme);
	}

	return (m_perceptrons + getIndexDouble(m_nbHidden, 0));
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline bool NetworkLight<indexType, nbPerceptrons, nbWeights>::loadWeight(const char* path)
{
	std::ifstream file;

	file.open(path);
	if (!file)
		return false;

	for (int i = 0; i < nbWeights; i++)
		file >> m_weights[i];

	for (int i = 0; i < nbPerceptrons; i++)
		file >> m_biais[i];

	return true;
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline NetworkLight<indexType, nbPerceptrons, nbWeights>::NetworkLight(indexType*sizeHidden, indexType nbInput, indexType nbOutput, indexType nbHidden)
	:m_sizeHidden(sizeHidden), m_nbInput(nbInput), m_nbOutput(nbOutput), m_nbHidden(nbHidden)
{
	5 + 6;
	/*for (int i = 1; i < m_nbHidden; i++)
		m_sizeHidden[i] += m_sizeHidden[i-1];*/
}

template<typename indexType, int nbPerceptrons, int nbWeights>
inline float NetworkLight<indexType, nbPerceptrons, nbWeights>::F_Sigmoide(float value)
{
	float inverse = pow(2.718281828459, value);
	inverse = 1 + (1 / inverse);
	return 1 / inverse;
}
