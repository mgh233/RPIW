//
// Created by 母国宏 on 2023/5/8.
//

#ifndef RECWARP_GLOBALWARP_H
#define RECWARP_GLOBALWARP_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "lsd/lsd.h"


using namespace std;
using namespace cv;

typedef pair<double, double> PDD;
typedef pair<PDD, PDD> Line;

/*
 * GlobalWarp类，执行算法第二大步骤，最终得到处理完毕的mesh顶点坐标
 */
class GlobalWarp {

private:
    Mat img;    // 待处理的图片
    vector<pair<int, int>> vertexes;    // mesh顶点
    vector<vector<Line>> line_segements;// 分割后的直线
    vector<Line> lines;                 // 直线


    // 通过lsd算法找出图中的直线
    void find_lines() {
        // 处理图片
        Mat img_gray;
        cvtColor(img, img_gray, CV_BGR2GRAY);
        auto * img_pixels = new double[img.cols * img.rows];
        for (int i = 0; i < img.rows; i ++) {
            for (int j = 0; j < img.cols; j ++) {
                img_pixels[i * img.cols + j] = img_gray.at<uchar>(i, j);
            }
        }

        double *out;
        int lines_number;
        out = lsd(&lines_number, img_pixels, img.cols, img.rows);
        // 提取直线信息
        for (int i = 0; i < lines_number; i ++) {
            PDD pixel_1 = make_pair(out[7 * i + 1], out[7 * i]);
            PDD pixel_2 = make_pair(out[7 * i + 3], out[7 * i + 2]);
            lines.push_back({pixel_1, pixel_2});
        }
    }


    // 通过mesh分割直线
    vector<vector<Line>> get_line_segment() {
        for (auto line: lines) {

        }
    }



public:
    GlobalWarp(vector<pair<int, int>> mesh, Mat img, int meshRow, int meshCol) {
        this->vertexes = mesh;
        this->img = img;
        this->line_segements = vector<vector<Line>>(meshRow, vector<Line>(meshCol));

        find_lines();
    }

};

#endif //RECWARP_GLOBALWARP_H
