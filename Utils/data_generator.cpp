#pragma once

#include <iostream>
#include <vector>
#include <random>

double generateRandomNumBetween(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

std::vector<std::vector<std::vector<double>>> dataGenerator(int s) {
    std::vector<std::vector<double>>inputs;
    std::vector<std::vector<double>> outputs;
    for (int i = 0; i < s; ++i) {
        double x = generateRandomNumBetween(0, 1);
        double y = x * x;
        inputs.push_back({ x });
        outputs.push_back({ y });
    }
    std::vector<std::vector<std::vector<double>>> v = { inputs,outputs };
    return v;
}