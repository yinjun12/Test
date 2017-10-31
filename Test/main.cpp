
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include<fstream>
#include<iostream>
#include<sstream>
#include <utility>
#include <vector>
#include <chrono>
#include<algorithm>
#include <memory>
#include <ctime>
#include <iomanip>
#include <iterator>
#include <future>
#include <set>
#include <algorithm>

#include "test/tools.h"
using namespace std;
#if 0
#define Data  165   /*��������*/
#define In 42  /*ÿ�������ж��ٸ��������*/
#define Out 1 /*ÿ�������ж��ٸ��������*/
#define Neuron 45 /*��Ԫ����*/
#define TrainC 20000  //ѵ������
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

/* d_in[Data][In] �洢 Data ��������ÿ�������� In �����롣
*d_out[Data][Out] �洢 Data ��������ÿ�������� Out �������
*w[Neuron][In]  ��ʾĳ�������ĳ����Ԫ��Ȩ�أ�v[Out][Neuron] ����ʾĳ����Ԫ��ĳ
*�������Ȩ�أ���֮��Ӧ�ı����������������������� dw[Neuron][In]
*�� dv[Out][Neuron]������ o[Neuron] ��¼������Ԫͨ���������
*��������OutputData[Out]  �洢BP������������
*/
double d_in[Data][In], d_out[Data][Out];
double w[Neuron][In], o[Neuron], v[Out][Neuron];
double Maxin[In], Minin[In], Maxout[Out], Minout[Out];
double OutputData[Out];
double dv[Out][Neuron], dw[Neuron][In];
double e;

/*���ļ���д��ʾ����������ʱ�ɲ���*/
void writeTest() {
	FILE *fp1, *fp2;
	double r1, r2;
	int i;
	srand((unsigned)time(NULL));
	if ((fp1 = fopen("G:\\in.txt", "w")) == NULL) {
		printf("can not open the in file\n");
		exit(0);
	}
	if ((fp2 = fopen("G:\\out.txt", "w")) == NULL) {
		printf("can not open the out file\n");
		exit(0);
	}


	for (i = 0; i<Data; i++) {
		r1 = rand() % 1000 / 100.0;
		r2 = rand() % 1000 / 100.0;
		fprintf(fp1, "%lf  %lf\n", r1, r2);
		fprintf(fp2, "%lf \n", r1 + r2);
	}
	fclose(fp1);
	fclose(fp2);
}

/*���ļ��ж�ȡ����*/
void readData() {
	fstream in;
	in.open("H:\\data.csv");
	string line = "";
	int dataNum = 0;
	while (dataNum<Data&&getline(in, line)) {
		replace(line.begin(), line.end(), ',', ' ');
		stringstream ss(line);
		int InCnt = 0;
		ss >> d_out[dataNum][0];
		while (InCnt<In&&ss >> d_in[dataNum][InCnt]) {
			InCnt++;
		}
		dataNum++;
	}
}

/*��ʼ��BP������*/
void initBPNework() {

	int i, j;

	for (i = 0; i<In; i++) {
		Minin[i] = Maxin[i] = d_in[0][i];
		for (j = 0; j<Data; j++)
		{
			Maxin[i] = Maxin[i]>d_in[j][i] ? Maxin[i] : d_in[j][i];
			Minin[i] = Minin[i]<d_in[j][i] ? Minin[i] : d_in[j][i];
		}
	}

	for (i = 0; i<Out; i++) {
		Minout[i] = Maxout[i] = d_out[0][i];
		for (j = 0; j<Data; j++)
		{
			Maxout[i] = Maxout[i]>d_out[j][i] ? Maxout[i] : d_out[j][i];
			Minout[i] = Minout[i]<d_out[j][i] ? Minout[i] : d_out[j][i];
		}
	}

	for (i = 0; i < In; i++)
		for (j = 0; j < Data; j++)
			d_in[j][i] = (d_in[j][i] - Minin[i] + 1) / (Maxin[i] - Minin[i] + 1);

	for (i = 0; i < Out; i++)
		for (j = 0; j < Data; j++)
			d_out[j][i] = (d_out[j][i] - Minout[i] + 1) / (Maxout[i] - Minout[i] + 1);

	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < In; ++j) {
			w[i][j] = rand()*2.0 / RAND_MAX - 1;
			dw[i][j] = 0;
		}

	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < Out; ++j) {
			v[j][i] = rand()*2.0 / RAND_MAX - 1;
			dv[j][i] = 0;
		}
}

/*����BP������Ԥ��� i �����������*/
void computO(int var) {

	int i, j;
	double sum, y;
	for (i = 0; i < Neuron; ++i) {
		sum = 0;
		for (j = 0; j < In; ++j)
			sum += w[i][j] * d_in[var][j];
		o[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < Out; ++i) {
		sum = 0;
		for (j = 0; j < Neuron; ++j)
			sum += v[i][j] * o[j];

		OutputData[i] = sum;
	}
}

/*�Ǹ���Ԥ��ĵ� i ������������������Ȩ�ؽ��и��£�e���������*/
void backUpdate(int var)
{
	int i, j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t = 0;
		for (j = 0; j < Out; ++j) {
			t += (OutputData[j] - d_out[var][j])*v[j][i];

			dv[j][i] = A*dv[j][i] + B*(OutputData[j] - d_out[var][j])*o[i];
			v[j][i] -= dv[j][i];
		}

		for (j = 0; j < In; ++j) {
			dw[i][j] = a*dw[i][j] + b*t*o[i] * (1 - o[i])*d_in[var][j];
			w[i][j] -= dw[i][j];
		}
	}
}

/*ѵ����Ϻ󣬸�������������õ�������*/
double result(double var1, double var2)
{
	int i, j;
	double sum, y;

	var1 = (var1 - Minin[0] + 1) / (Maxin[0] - Minin[0] + 1);
	var2 = (var2 - Minin[1] + 1) / (Maxin[1] - Minin[1] + 1);

	for (i = 0; i < Neuron; ++i) {
		sum = 0;
		sum = w[i][0] * var1 + w[i][1] * var2;
		o[i] = 1 / (1 + exp(-1 * sum));
	}
	sum = 0;
	for (j = 0; j < Neuron; ++j)
		sum += v[0][j] * o[j];

	return sum*(Maxout[0] - Minout[0] + 1) + Minout[0] - 1;
}

double result(double input[]) {
	int i, j;
	double sum, y;
	for (i = 0; i < In; i++) {
		input[i] = (input[i] - Minin[i] + 1) / (Maxin[i] - Minin[i] + 1);
	}

	for (i = 0; i < Neuron; ++i) {
		sum = 0;
		for (j = 0; j < In; j++) {
			sum += w[i][j] * input[j];
		}
		o[i] = 1 / (1 + exp(-1 * sum));
	}

	sum = 0;
	for (j = 0; j < Neuron; ++j)
		sum += v[0][j] * o[j];

	return sum*(Maxout[0] - Minout[0] + 1) + Minout[0] - 1;
}

/*����Ԫ�����������*/
void writeNeuron()
{
	FILE *fp1;
	int i, j;
	if ((fp1 = fopen("G:\\neuron.txt", "w")) == NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < In; ++j) {
			fprintf(fp1, "%lf ", w[i][j]);
		}
	fprintf(fp1, "\n\n\n\n");

	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < Out; ++j) {
			fprintf(fp1, "%lf ", v[j][i]);
		}

	fclose(fp1);
}

void readNeuron() {
	FILE *fp1;
	int i, j;
	if ((fp1 = fopen("G:\\neuron.txt", "r")) == NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	/*printf("w:\n");*/
	for (i = 0; i < Neuron; ++i)
	{
		/*printf("%d:", i);*/
		for (j = 0; j < In; ++j) {
			fscanf(fp1, "%lf ", &w[i][j]);
			/*printf("%lf ", w[i][j]);*/
		}
		/*printf("\n");*/
	}
	/*printf("v:\n");*/
	for (i = 0; i < Neuron; ++i)
		for (j = 0; j < Out; ++j) {
			fscanf(fp1, "%lf ", &v[j][i]);
		}

	fclose(fp1);
}

void writeMinAndMax() {
	FILE *fp1;
	int i, j;
	if ((fp1 = fopen("G:\\min_max.txt", "w")) == NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}

	for (int i = 0; i < In; i++) {
		fprintf(fp1, "%lf ", Minin[i]);
	}
	fprintf(fp1, "\n\n\n\n");

	for (int i = 0; i < In; i++) {
		fprintf(fp1, "%lf ", Maxin[i]);
	}
	fprintf(fp1, "\n\n\n\n");

	for (int i = 0; i < Out; i++) {
		fprintf(fp1, "%lf ", Minout[i]);
	}
	fprintf(fp1, "\n\n\n\n");

	for (int i = 0; i < Out; i++) {
		fprintf(fp1, "%lf ", Maxout[i]);
	}

	fclose(fp1);
}

void readMinAndMax() {
	FILE *fp1;
	int i, j;
	if ((fp1 = fopen("G:\\min_max.txt", "r")) == NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}

	for (int i = 0; i < In; i++) {
		fscanf(fp1, "%lf ", &Minin[i]);
	}

	for (int i = 0; i < In; i++) {
		fscanf(fp1, "%lf ", &Maxin[i]);
	}

	for (int i = 0; i < Out; i++) {
		fscanf(fp1, "%lf ", &Minout[i]);
	}

	for (int i = 0; i < Out; i++) {
		fscanf(fp1, "%lf ", &Maxout[i]);
	}

	fclose(fp1);
}
/*ѵ��������*/
void  trainNetwork() {

	int i, c = 0, j;
	do {
		e = 0;
		for (i = 0; i < Data; ++i) {
			computO(i);
			for (j = 0; j < Out; ++j)
				e += fabs((OutputData[j] - d_out[i][j]) / d_out[i][j]);
			backUpdate(i);
		}
		printf("ѵ��������%d  ���ȣ�%lf\n", c, e / Data);
		c++;
	} while (c<TrainC && e / Data>0.01);

	printf("**********************************\n");
	printf("*����С��0.01,���Ͼ���Ҫ��*\n");
	printf("**********************************\n");
}
#endif
std::vector<std::promise<int>>pVec;

std::ostream& operator<<(std::ostream& out, const std::vector<int>& vec) {
	for (auto& it : vec) {
		out << it << " ";
	}
	return out;
}

struct Person {
	int a;
	std::string str;
};
void setShared( std::shared_ptr<Person> p) {
	p = std::make_shared<Person>();
	p->a = 5;
	p->str = "test";
}
int  main(int argc, char const *argv[])
{
	/*readData();
	initBPNework();
	trainNetwork();*/

	/*Ԥ����*/
	/*readNeuron();
	readMinAndMax();
	double input1[] = {1,	5,	2,	3,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,
		0,	0,	0,	1,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	1,	1,	0,
		0,	0,	0,	0,	1,	0,	0,	0,
	};
	printf("%lf \n", result(input1));

	double input2[] = { 2,	6,	4,	3,	1,	0,	0,	0,	0,	1,	1,	0,	1,	0,	0,	0,
		0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	1,	1,	0,
		0,	0,	0,	0,	1,	0,
	};

	double input3[] = { 2,	5,	1,	3,	3,	0,	0,	0,	0,	1,	1,	0,	1,	0,	1,	0,
		1,	0,	0,	0,	1,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	1,	0,	0,
		0,	0,	0,	0,	1,	0,
	};
	printf("%lf \n", result(input2));
	printf("%lf \n", result(input3));*/
	///*��ѵ��������Ԫд���ļ���*/
	std::set<int>s;
	s.insert(4);
	s.insert(3);
	s.insert(5);
	auto it = s.find(3);
	s.erase(it);
	s.insert(6);
	for (auto it : s) {
		std::cout << it << " ";
	}
}