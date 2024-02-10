#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

static std::vector<std::vector<double>> parseCSV(std::string filepath) {
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



#endif // !LOAD_CSV_H
