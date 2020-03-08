/*
deepNetworkMini.h

int m_nb is for debuging

Create by
MATHGIQUE
*/

#ifndef NETWORK_MINI
#define NETWORK_MINI

#include <vector>
#include <string>

class NetworkMini
{
private:
	std::vector<std::vector<std::vector<double>>> m_weights;
	std::vector< std::vector<double>> m_biais;
	std::vector< std::vector<double>> m_perceptrons;
	std::vector< std::vector<double>> m_delta;
	std::vector<double> *m_input;
	std::vector<double> *m_excepted;
	int m_nbInput;
	int m_nbOutput;
	int m_nb;
	double m_coefficient;
	std::string m_fLogi;

	void feedForward();
	void backPropagation();
	void setInput(std::vector<double> *input);
	void setExcepted(std::vector<double> *excepted);
	
public:
	NetworkMini(int inputLayer, int outputLayer, std::vector<int> &hiddenLayer, std::string &f);
	void runLearning(std::vector<std::vector<double>> *test, std::vector<std::vector<double>> *resultDesired);
	void runPrediction(std::vector<double> *data);
	bool setCoefficient(double value);
	static double F_Sigmoide(double value);
	static double F_Prime(double value);
	static double F_Tangente(double value);
	static void F_Softmax(std::vector<double> *value);
};

#endif