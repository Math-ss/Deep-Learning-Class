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
#include"Include/deepNetworkMini.h"
#include"Include/main.h"

using namespace std;

int main(void)
{
  srand(time(NULL));

  ifstream Ftrain("");

  if (Ftrain)
	  printf("cool\n");

  vector<int> define(1, 15);
  vector<vector<double>> train(100, vector<double>(784));
  vector<vector<double>> prevu(100, vector<double>(1, 0));
  vector<int> visi(10);

  for (int i = 0; i < 100; i++)
  {
	  double toutou = 6.0;
	  char passe;
	  Ftrain >> toutou;
	  Ftrain >> passe;

	  if (toutou == 9)
		  prevu[i][0] = 1;

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

  NetworkMini numberImage(784, 1, define, string("sigmoide"));

  printf("cool>Bis\n");
  

  for(int i = 0; i < 100'000; i++)
	numberImage.runLearning(&train, &prevu);

  system("pause");

  return 0;
}