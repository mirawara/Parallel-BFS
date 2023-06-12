#include<fstream>
#include<iostream>
#include<cmath>
#include<ctime>
#include<cstdlib>

#define RAND_MAX 10000
using namespace std;

int RAND_MAXX;
void generate(char *s, int max)
{
	
	ofstream fout;
	srand(std::time(0));	
	fout.open(s);
	RAND_MAXX = max;
	fout << RAND_MAXX << "," << endl;
	for(int i=1;i<=RAND_MAXX;i++)
	{
		fout << i;
		int temp = rand()%100;
		for(int j=1;j<=temp;j++)
		{
			int temp2 = 0;
			while(!temp2)
			{
				temp2 = rand()%RAND_MAXX;
			}
			fout << "," << temp2;
		}
		fout << endl;
	}
	fout.close();
}

int main()
{
	generate("graph1000.txt", 1000000);
	//generate("graph60.txt", 60000);
	//generate("graph70.txt", 70000);
	//generate("graph80.txt", 80000);
	//generate("graph90.txt", 90000);
	
	return 0;
}


