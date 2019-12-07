/*
main.cpp

Example how use the NetworkMini class
It will learn to find all 9 in the MNIST (100th to 200th)

Create by
MATHGIQUE
*/

#include<vector>
#include<fstream>
#include<cstdlib>
#include<ctime>
#include<cstdio>
#include"..\Include\deepNetworkMini.h"
#include"..\Include\main.h"

using namespace std;

int main(void)
{
  srand(time(NULL));

  ifstream Ftrain("D:/Users/Ma-Game/Documents/Mathis/Programmation/C++/IA/mnist_train_complete.csv");

  if (Ftrain)
	  printf("cool\n");

  vector<int> define(1, 15);
  vector<vector<double>> train(200, vector<double>(784));
  vector<vector<double>> prevu(200, vector<double>(10, 0));
  vector<int> visi(10);
  int decide = 0;
  vector<double> test(784);

  for (int i = 0; i < 200; i++)
  {
	  double toutou = 6.0;
	  char passe;
	  Ftrain >> toutou;
	  Ftrain >> passe;

	  prevu[i][toutou] = 1;

	  visi[toutou]++;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> toutou;
		  Ftrain >> passe;
		  toutou /= (255 * 10);
		  train[i][j] = toutou;
	  }
	  Ftrain >> toutou;
	  toutou /= (255 * 10);
	  train[i][783] = toutou;
  }

  for (int i = 0; i < 1; i++)
  {
	  double toutou = 6.0;
	  char passe;
	  Ftrain >> toutou;
	  Ftrain >> passe;

	  decide = toutou;

	  for (int j = 0; j < 783; j++)
	  {
		  Ftrain >> toutou;
		  Ftrain >> passe;
		  toutou /= (255 * 10);
		  test[j] = toutou;
	  }
	  Ftrain >> toutou;
	  toutou /= (255 * 10);
	  test[783] = toutou;
  }

  NetworkMini numberImage(784, 10, define, string("sigmoide"));

  printf("cool>Bis\n");
  

  for(int i = 0; i < 5'000; i++)
	numberImage.runLearning(&train, &prevu);

  numberImage.runPrediction(&test);

  printf("%d", decide);
  system("pause");

  return 0;
}
