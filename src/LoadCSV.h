#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

static std::vector<std::vector<double>> ParseCSV(std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Failed to open dataset!" << std::endl;
    }
    std::string line;
    std::vector<std::vector<double>> parsed_CSV;

    while (std::getline(file, line)) {
        std::vector<double> parsed_line;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            parsed_line.push_back(std::stod(line.substr(0, pos)));
            line.erase(0, pos + 1);
        }
        parsed_line.push_back(std::stod(line));
        parsed_CSV.push_back(parsed_line);
    }

    file.close();

    return parsed_CSV;
} 

static std::vector<double> GetGroundTruth(const std::vector<double>& image) {
    std::vector<double> results;
    results.resize(10);
    std::fill(results.begin(), results.end(), 0);
    results[image[0]] = 1;
    return results;
}



#endif // !LOAD_CSV_H
