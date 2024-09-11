#pragma once

#include "data_generator.h"
#include "print_vec.h"
#include "MLP.h"

#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

#include "YES_NO_ENUM.cpp"

using namespace std;

vector<vector<vector<double>>> readDataFromFile(string file_name)
{
    ifstream file("./train.txt");
    if (!file.is_open())
    {
        throw invalid_argument("File not found");
    }
    vector<vector<double>> inputs;
    vector<vector<double>> desired;
    string line;
    while (getline(file, line))
    {
        vector<double> input;
        vector<double> output;
        stringstream ss(line);
        string token;
        bool is_input = true;
        while (getline(ss, token, ','))
        {
            if (is_input)
            {
                input.push_back(stod(token));
            }
            else
            {
                output.push_back(stod(token));
            }
            is_input = !is_input;
        }
        inputs.push_back(input);
        desired.push_back(output);
    }
    return {inputs, desired};
}

int main()
{
    try
    {
        int total_presentations = 100;

        vector<vector<vector<double>>> vec = dataGenerator(total_presentations);
        vector<vector<double>> inputs = vec[0];
        vector<vector<double>> desired = vec[1];

        // vector<vector<double>> inputs = { {0.1}, {0.2}, {0.5} };
        // vector<vector<double>> desired = { {0.01}, {0.02}, {0.25} };

        vector<int> nodes_per_layer = {1, 4, 1};
        int size = nodes_per_layer.size();

        MLP mlp(size, nodes_per_layer);

        mlp.set_learning_const(0.5);
        mlp.set_error_limit(0.01);
        char ch;
        cout << "Do want to print the error after each presentation? (y/n): ";
        cin >> ch;
        if (ch == 'y')
        {
            mlp.set_error_percent_print_mode(YES);
        }
        else
        {
            mlp.set_error_percent_print_mode(NO);
        }
        cout << "Do you want to initially training of the network? (y/n): ";
        cin >> ch;
        if (ch == 'y')
        {
            cout << "Training started\n";
            mlp.train(total_presentations, inputs, desired);
            cout << "Training completed\n";
        }
        while (true)
        {
            // user panel to ask what to do
            cout << "Select an option: \n";
            cout << "1. Test the network\n";
            cout << "2. Train the network\n";
            cout << "3. Save the model to a File\n";
            cout << "4. Read the model from a File\n";
            cout << "5. Exit\n";
            int option;
            cin >> option;
            if (option == 1)
            {
                cout << "Enter the input to test the network: ";
                double input;
                cin >> input;
                vector<double> test_input = {input};
                vector<double> output = mlp.predict(test_input);
                cout << "Output: " << output[0] << endl;
            }
            else if (option == 2)
            {
                cout << "Do have a file with training data? (y/n): ";
                char ch;
                cin >> ch;
                if (ch == 'y')
                {
                    cout << "Enter the file name: ";
                    string file_name;
                    cin >> file_name;
                    cout << "Reading data from file\n";
                    vec = readDataFromFile(file_name);
                    inputs = vec[0];
                    desired = vec[1];
                    total_presentations = inputs.size();
                }
                else
                {
                    inputs.clear();
                    desired.clear();
                    cout << "Enter the number of presentations: ";
                    int total_presentations;
                    cin >> total_presentations;
                    cout << "Do you want to generate random data for training? (y/n): ";
                    char choice;
                    cin >> choice;
                    if (choice == 'y')
                    {
                        vec = dataGenerator(total_presentations);
                        inputs = vec[0];
                        desired = vec[1];
                    }
                    else
                    {
                        inputs.clear();
                        desired.clear();
                        for (int i = 0; i < total_presentations; i++)
                        {
                            cout << "Enter the input for presentation " << i + 1 << ": ";
                            double input;
                            cin >> input;
                            inputs.push_back({input});
                            cout << "Enter the desired output for presentation " << i + 1 << ": ";
                            double output;
                            cin >> output;
                            desired.push_back({output});
                        }
                    }
                }
                mlp.train(total_presentations, inputs, desired);
            }
            else if (option == 5)
            {
                break;
            }
            else
            {
                cout << "Invalid option selected\n";
            }
        }
    }
    catch (exception e)
    {
        cout << e.what();
    }
    return 0;
}
