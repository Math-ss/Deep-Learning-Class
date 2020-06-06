/*
main.cpp

Example how use the BackPropagationLearning class
It will learn to class the numbers(0th to 200th)

Created by Math's
*/

#include<vector>
#include<fstream>
#include<cstdlib>
#include<ctime>
#include<cstdio>
#include "BackPropagationLearning.h"
#include<iostream>

using namespace std;

int main(void)
{
  srand(time(NULL));

  ifstream Ftrain("mnist_train_complete.csv");

  if (!Ftrain)
	  return 1;
  
  //Contain size of each hidden layer
  vector<int> define(1, 15);

  //Training Data
  vector<vector<float>> train(200, vector<float>(784));
  vector<vector<float>> prevu(200, vector<float>(10, 0));

  //For the test
  vector<int> visi(10);
  int decide = 0;
  vector<float> test(784);
 

  for (int i = 0; i < 200; i++)
  {
	  float buffer = 6.0f;
	  char passe;
	  Ftrain >> buffer;
	  Ftrain >> passe;

	  prevu[i][buffer] = 1;

	  visi[buffer]++;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> buffer;
		  Ftrain >> passe;
		  buffer /= (255.0f * 10.0f);
		  train[i][j] = buffer;
	  }
	  Ftrain >> buffer;
	  buffer /= (255.0f * 10.0f);
	  train[i][783] = buffer;
  }

  {
	  float buffer = 6.0f;
	  char passe;
	  Ftrain >> buffer;
	  Ftrain >> passe;

	  decide = (int) buffer;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> buffer;
		  Ftrain >> passe;
		  buffer /= (255.0f * 10.0f);
		  test[j] = buffer;
	  }
	  Ftrain >> buffer;
	  buffer /= (255.0f * 10.0f);
	  test[783] = buffer;
  }

  string fonctiun("sigmoide");

  //Setup the network
  BackPropagationLearning MNIST(784, 10, define, fonctiun);
  MNIST.setData(train);
  MNIST.setExcepted(prevu);

  //Train the Newtork on data : will compute 2'500 times the complete std::vector
  MNIST.training(2'500 * train.size());

  //Compute prediction for test and get the results
  MNIST.computePrediction(test);
  const vector<float>& result = MNIST.getPredictionCopy();

  //Save wheights (could be used with NetworkLight)
  MNIST.saveWeights("");

  system("pause");

  return 0;
}
