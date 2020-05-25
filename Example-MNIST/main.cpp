/*
main.cpp

Example how use the NetworkMini class
It will learn to class the numbers(0th to 200th)

Create by
MATHGIQUE
*/

#include<vector>
#include<fstream>
#include<cstdlib>
#include<ctime>
#include<cstdio>
#include"deepNetworkMini.h"
#include <iostream>

using namespace std;

int main(void)
{
  srand(time(NULL));

  ifstream Ftrain("mnist_train_complete.csv");

  if (!Ftrain)
	  return 1;
  
  vector<int> define(1, 15);
  vector<vector<double>> train(200, vector<double>(784));
  vector<vector<double>> prevu(200, vector<double>(10, 0));
  vector<int> visi(10);
  int decide = 0;
  vector<double> test(784);

  for (int i = 0; i < 200; i++)
  {
	  double buffer = 6.0;
	  char passe;
	  Ftrain >> buffer;
	  Ftrain >> passe;

	  prevu[i][buffer] = 1;

	  visi[buffer]++;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> buffer;
		  Ftrain >> passe;
		  buffer /= (255 * 10);
		  train[i][j] = buffer;
	  }
	  Ftrain >> buffer;
	  buffer /= (255 * 10);
	  train[i][783] = buffer;
  }

  {
	  double buffer = 6.0;
	  char passe;
	  Ftrain >> buffer;
	  Ftrain >> passe;

	  decide = (int) buffer;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> buffer;
		  Ftrain >> passe;
		  buffer /= (255 * 10);
		  test[j] = buffer;
	  }
	  Ftrain >> buffer;
	  buffer /= (255 * 10);
	  test[783] = buffer;
  }

  string fonctiun("sigmoide");

  NetworkMini MNIST(784, 10, define, fonctiun);
  
  MNIST.setCoefficient(0.5);//I've just a little tried(5 times). I'm sure it's possible to do better

  for(int i = 0; i < 2'500; i++)
	  MNIST.runLearning(&train, &prevu);

  MNIST.runPrediction(&test);
  MNIST.saveWeights("");

  system("pause");

  return 0;
}
