#ifndef FILE_READER_H
#define FILE_READER_H

#include <fstream>
#include <vector>
#include <sstream>

#include "dval.h"
#include "matrix.h"

struct Image {
    Matrix<Dval> label;
    Matrix<Dval> image_data;
};

static Matrix<Dval> OneHotEncode(int label) {
    Matrix<Dval> ret(10, 1);
    std::vector<Dval> elements;
    elements.resize(10);
    for (auto& element : elements)
        element = 0;
    elements[label] = 1;

    ret.SetElements(elements);
    return ret;
}
static std::vector<Image> ReadData(const std::string& filepath,
                                   int rows, int batch_size) {
    std::ifstream file(filepath);
    if (!file.is_open())
        std::cout << "File didn't open.\n";
    std::vector<Image> images;
    images.reserve(rows / batch_size);
    std::string line;
    for (int i = 0; i < rows / batch_size; i++) {
        std::vector<Dval> label_data;
        std::vector<Dval> image_data;
        for (int j = 0; j < batch_size; j++) {
            getline(file, line);
            int label = std::stoi(line.substr(0, 1));
            for (int k = 0; k < 10; k++) {
                if (k == label)
                    label_data.emplace_back(1);
                else 
                    label_data.emplace_back(0);
            }
            line.erase(0, 2);

            std::istringstream ss(line);
            std::string value;
            while (getline(ss, value, ',')) {
                image_data.emplace_back((double)std::stoi(value) / 255);
            }
        }
        Image image;
        image.label.row_count_ = batch_size;
        image.label.col_count_ = 10;
        image.label.SetElements(label_data);
        image.image_data.row_count_ = batch_size;
        image.image_data.col_count_ = 784;
        image.image_data.SetElements(image_data);
        image.label = image.label.Transpose();
        image.image_data = image.image_data.Transpose();
        images.push_back(image);
    }
    return images;
}

#endif // !FILE_READER_H
