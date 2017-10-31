#pragma once
#include <iostream>
#include <utility>
#include <vector>
#include<memory>
#include <string>
#include<assert.h>
#include<map>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<stdlib.h>
#include<stdio.h>
#include<sstream>
#include<fstream>

#include <boost/lexical_cast.hpp>
#include<boost/enable_shared_from_this.hpp>

using namespace std;

#define Data  820
#define In 2
#define Out 1
#define Neuron 45
#define TrainC 5500

#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3
//class Test : public enable_shared_from_this<Test>
//{
//public:
//	shared_ptr<Test> getTestPtr()
//	{
//		return shared_from_this();
//	}
//};

//void transformIntegerToChineseFormt(int num)
//{
//	std::vector<std::string> numForm{ "零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖" };
//	std::vector<std::string>unit{ "元", "拾", "佰", "仟", "万",
//		// 拾万位到仟万位  
//		"拾", "佰", "仟",
//		// 亿位到万亿位  
//		"亿", "拾", "佰", "仟", "万" };
//	std::string numStr = to_string(num);
//	std::string result = "";
//	for (int i = 0; i<numStr.size(); i++)
//	{
//		int index = numStr[i] - '0';
//		result += numForm[index] + unit[numStr.size() - i - 1];
//	}
//	std::vector<std::string>rep{ "零拾","零佰","零仟","零万","零零零零","零零零","零零" };
//	for (auto str : rep) {
//		size_t index = result.find(str);
//		if (index != result.npos)
//			result.replace(index, str.length(), "零");
//	};
//	std::cout << result << std::endl;
//}
//int Minin[In], Minout[Out], Maxin[In], Maxout[Out];
//int d_in[Data][In], d_out[Data][Out];
//int w[Neuron][In], v[Out][Neuron], dw[Neuron][In], dv[Out][Neuron], o[Neuron], OutputData[Out];
//
//
//void initBPNework() {
//
//	int i, j; /*找到数据最小、最大值*/
//	for (i = 0; i<In; i++) {
//		Minin[i] = Maxin[i] = d_in[0][i];
//		for (j = 0; j<Data; j++)
//		{
//			Maxin[i] = Maxin[i]>d_in[j][i] ? Maxin[i] : d_in[j][i];
//			Minin[i] = Minin[i]<d_in[j][i] ? Minin[i] : d_in[j][i];
//		}
//	}
//	for (i = 0; i<Out; i++) {
//		Minout[i] = Maxout[i] = d_out[0][i];
//		for (j = 0; j<Data; j++)
//		{
//			Maxout[i] = Maxout[i]>d_out[j][i] ? Maxout[i] : d_out[j][i];
//			Minout[i] = Minout[i]<d_out[j][i] ? Minout[i] : d_out[j][i];
//		}
//	}
//	/*
//	　　　　归一化处理
//		　　　　*/
//	for (i = 0; i < In; i++)
//		for (j = 0; j < Data; j++)
//			d_in[j][i] = (d_in[j][i] - Minin[i] + 1) / (Maxin[i] - Minin[i] + 1);
//	for (i = 0; i < Out; i++)
//		for (j = 0; j < Data; j++)
//			d_out[j][i] = (d_out[j][i] - Minout[i] + 1) / (Maxout[i] - Minout[i] + 1);
//	/*
//	　　　　初始化神经元
//		　　*/
//	for (i = 0; i < Neuron; ++i)
//		for (j = 0; j < In; ++j) {
//			w[i][j] = (rand()*2.0 / RAND_MAX - 1) / 2;
//			dw[i][j] = 0;
//		}
//	for (i = 0; i < Neuron; ++i)
//		for (j = 0; j < Out; ++j) {
//			v[j][i] = (rand()*2.0 / RAND_MAX - 1) / 2;
//			dv[j][i] = 0;
//		}
//}
//
//void  trainNetwork() {
//	int i, c = 0;
//	do {
//		e = 0;
//		for (i = 0; i < Data; ++i) {
//			computO(i);
//			e += fabs((OutputData[0] - d_out[i][0]) / d_out[i][0]);
//			backUpdate(i);
//		}
//		//printf("%d  %lf\n",c,e/Data);
//		c++;
//	} while (c<TrainC && e / Data>0.01);
//}
//
//void computO(int var) {
//
//	int i, j;
//	double sum, y;
//
//	/*
//	神经元输出
//	*/
//
//	for (i = 0; i < Neuron; ++i) {
//		sum = 0;
//		for (j = 0; j < In; ++j)
//			sum += w[i][j] * d_in[var][j];
//		o[i] = 1 / (1 + exp(-1 * sum));
//	}
//
//	/*  隐藏层到输出层输出 */
//
//	for (i = 0; i < Out; ++i) {
//		sum = 0;
//		for (j = 0; j < Neuron; ++j)
//			sum += v[i][j] * o[j];
//
//		OutputData[i] = sum;
//	}
//}
//
//void backUpdate(int var)
//{
//	int i, j;
//	double t;
//	for (i = 0; i < Neuron; ++i)
//	{
//		t = 0;
//		for (j = 0; j < Out; ++j) {
//			t += (OutputData[j] - d_out[var][j])*v[j][i];
//
//			dv[j][i] = A*dv[j][i] + B*(OutputData[j] - d_out[var][j])*o[i];
//			v[j][i] -= dv[j][i];
//		}
//
//		for (j = 0; j < In; ++j) {
//			dw[i][j] = a*dw[i][j] + b*t*o[i] * (1 - o[i])*d_in[var][j];
//			w[i][j] -= dw[i][j];
//		}
//	}
//}