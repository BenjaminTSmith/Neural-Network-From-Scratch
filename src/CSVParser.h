#ifndef CSV_PARSE_H
#define CSV_PARSE_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

struct Image {
    int label = 0;
    std::vector<double> data;
    std::vector<double> one_hot_label; 
};

static std::vector<double> GetGroundTruth(int label) {
    std::vector<double> results;
    results.resize(10);
    std::fill(results.begin(), results.end(), 0);
    results[label] = 1;
    return results;
}

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
            parsed_line.push_back(std::stod(line.substr(0, pos)));
            line.erase(0, pos + 1);
        }
        parsed_line.push_back(std::stod(line));

        int label = parsed_line[0];
        parsed_line.erase(parsed_line.begin());
        parsed_CSV.push_back({label, parsed_line});
    }
    for (auto& image : parsed_CSV) {
        for (auto& data : image.data) {
            data /= 255;
        }
        //image.one_hot_label = OneHotEncode(image.label, 10);
    }

    file.close();

    return parsed_CSV;
} 




#endif // !CSV_PARSE_H
