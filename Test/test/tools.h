#pragma once
#include <iostream>
#include <string>
#include <random>

#define PI (3.1415926)

double randomAngle() {
	std::random_device rd;
	std::default_random_engine engine1(rd());
	std::uniform_int_distribution<>dis(0, 360);
	int tmpInt = dis(engine1);
	double ret = (static_cast<double>(tmpInt)*PI) / 180;
	return ret;
}

int generateIntRandom(int start, int end) {
	std::random_device rd;
	std::default_random_engine engine(rd());
	std::uniform_int_distribution<>dis(start, end);
	return dis(engine);
}

double generateDoubleRandom(double start, double end) {
	std::random_device rd;
	std::default_random_engine engine(rd());
	std::uniform_real_distribution<>dis(start, end);
	return dis(engine);
}
