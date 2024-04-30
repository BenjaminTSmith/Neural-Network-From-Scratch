#ifndef FILE_READER_H
#define FILE_READER_H

#include <fstream>
#include <vector>

#include "dval.h"
#include "matrix.h"

struct Image {
    unsigned char label;
    Matrix<Dval> image_data;
};

static std::vector<Image> ReadDataset(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open())
        std::cout << "File didn't open.\n";
    std::vector<Image> images;
    images.reserve(60000);
    std::string line;
    while (getline(file, line)) {
        Image image;
        image.label = std::stoi(line.substr(0, 1));
        line.erase(0, 2);

        images.push_back(image);
    }

    return images;
}

#endif // !FILE_READER_H
