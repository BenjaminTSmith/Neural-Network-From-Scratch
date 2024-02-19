#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>

struct Image {
    int label = 0;
    std::vector<double> data;
};

static std::vector<Image> ParseCSV(std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Failed to open dataset!" << std::endl;
    }
    std::string line;
    std::vector<Image> parsed_CSV;

    while (std::getline(file, line)) {
        std::vector<double> parsed_line;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos) {
            parsed_line.push_back(std::stod(line.substr(0, pos)) / 255);
            line.erase(0, pos + 1);
        }
        parsed_line.push_back(std::stod(line));

        int label = parsed_line[0] * 255;
        parsed_line.erase(parsed_line.begin());
        parsed_CSV.push_back({label, parsed_line});
    }

    file.close();

    return parsed_CSV;
} 

static std::vector<double> GetGroundTruth(int label) {
    std::vector<double> results;
    results.resize(10);
    std::fill(results.begin(), results.end(), 0);
    results[label] = 1;
    return results;
}



#endif // !LOAD_CSV_H
