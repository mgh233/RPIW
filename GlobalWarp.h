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
    int meshRow;
    int meshCol;
    Mat img;    // 待处理的图片
    vector<pair<int, int>> vertexes;    // mesh顶点
    vector<vector<vector<Line>>> line_segements;// 分割后的直线
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


    // 判断点是否在直线右侧
    bool isCordInRight(PDD cord, PDD p1, PDD p2) {
        double x = cord.first;
        double y = cord.second;

        double A = p2.second - p1.second;
        double B = p1.first - p2.first;
        double C = p1.second * (p2.first - p1.first) - p1.first * (p2.second - p1.second);
        return A * x + B * y + C >= 0;
    }


    // 判断坐标是否在quad中
    bool isCordInQuad(PDD cord, int quadRow, int quadCol) {
        int index = quadRow * (meshCol + 1) + quadCol;
        PDD topLeft = vertexes[index];
        PDD topRight = vertexes[index + 1];
        PDD bottomLeft = vertexes[index + (meshCol + 1)];
        PDD bottomRight = vertexes[index + (meshCol + 1) + 1];

        return isCordInRight(cord, topLeft, topRight) && isCordInRight(cord, topRight, bottomRight)
                && isCordInRight(cord, bottomRight, bottomLeft) && isCordInRight(cord, bottomLeft, topLeft);

    }


    // 得到两条直线的交点
    PDD getIntersection(PDD p1_1, PDD p1_2, PDD p2_1, PDD p2_2) {
        double x;
        double y;

        PDD result(-1,-1);

        double k = (p1_1.second - p1_2.second) / (p1_1.first - p1_2.first);
        double b = p1_1.second - k * p1_1.first;

        double k1 = (p2_1.second - p2_2.second) / (p2_1.first - p2_2.first);
        double b1 = p2_1.second - k1 * p2_1.first;

        if (fabs(p1_1.second - p1_2.second) < 0.001)
            k = 0, b = (p1_1.second + p1_2.second) / 2;
        if (fabs(p2_1.second - p2_2.second) < 0.001)
            k = 0, b = (p2_1.second + p2_2.second) / 2;

        if (fabs(p1_1.first - p1_2.first) < 0.001) {
            x = (p1_1.first + p1_2.first) / 2;
            y = k1 * x + b1;
        }
        else if (fabs(p2_1.first - p2_2.first) < 0.001) {
            x = (p2_1.first + p2_2.first) / 2;
            y = k * x + b;
        }
        else {
            x = (b1 - b) / (k - k1);
            y = k * x + b;
        }
        result.first = x;
        result.second = y;

        return result;
    }


    // 判断点是否在直线范围内
    bool isPointOnLine(PDD point, PDD p1, PDD p2) {
        double x = point.first;
        double y = point.second;
        return (x <= (p1.first > p2.first ? p1.first : p2.first)) && (x >= (p1.first < p2.first ? p1.first : p2.first)) &&
        (y <= (p1.second > p2.second ? p1.second : p2.second)) && (y >= (p1.second < p2.second ? p1.second : p2.second));
    }


    // 判断交点是否在直线上
    bool isIntersectionOnLine(Line line, PDD p1, PDD p2) {
        PDD intersection = getIntersection(line.first, line.second, p1, p2);
        return isPointOnLine(intersection, line.first, line.second) && isPointOnLine(intersection, p1, p2);
    }


    // 判断直线是否有部分在quad中
    bool isLineInQuad(Line line, int quadRow, int quadCol) {
        PDD line_start = line.first;
        PDD line_end = line.second;

        int index = quadRow * (meshCol + 1) + quadCol;
        PDD topLeft = vertexes[index];
        PDD topRight = vertexes[index + 1];
        PDD bottomLeft = vertexes[index + (meshCol + 1)];
        PDD bottomRight = vertexes[index + (meshCol + 1) + 1];

        // 如果任意一点在quad中即为真
        if (isCordInQuad(line_start, quadRow, quadCol) || isCordInQuad(line_end, quadRow, quadCol))
            return true;
        else {
            return isIntersectionOnLine(line, topLeft, topRight) || isIntersectionOnLine(line, topRight, bottomRight) ||
                    isIntersectionOnLine(line, bottomRight, bottomLeft) || isIntersectionOnLine(line, bottomLeft, topLeft);
        }
    }


    // 通过mesh分割直线
    void get_line_segment() {
        // 遍历每个quad，寻找能够分割的直线
        for (int i = 0; i < meshRow; i ++) {
            vector<vector<Line>> lineSegThisRow;
            for (int j = 0; j < meshCol; j ++) {
                vector<Line> linesSeg;  // 每个quad中的直线
                for (auto line: lines) {
                    // 如果该直线有部分在quad中则计算其分割量
                    if (isLineInQuad(line, i, j)) {
                        PDD line_start = line.first;
                        PDD line_end = line.second;

                        int index = i * (meshCol + 1) + j;
                        PDD topLeft = vertexes[index];
                        PDD topRight = vertexes[index + 1];
                        PDD bottomLeft = vertexes[index + (meshCol + 1)];
                        PDD bottomRight = vertexes[index + (meshCol + 1) + 1];

                        // 若两点都在quad内
                        if (isCordInQuad(line_start, i, j) && isCordInQuad(line_end, i, j)) {
                            linesSeg.push_back(line);
                        }
                        // 只有一点在quad内
                        else  {
                            // 判断交点是否满足条件
                            vector<PDD> point_final;
                            if (isIntersectionOnLine(line, topLeft, topRight))
                                point_final.push_back(getIntersection(line_start, line_end, topLeft, topRight));
                            if (isIntersectionOnLine(line, topRight, bottomRight))
                                point_final.push_back(getIntersection(line_start, line_end, topRight, bottomRight));
                            if (isIntersectionOnLine(line, bottomRight, bottomLeft))
                                point_final.push_back(getIntersection(line_start, line_end, bottomRight, bottomLeft));
                            if (isIntersectionOnLine(line, bottomLeft, topLeft))
                                point_final.push_back(getIntersection(line_start, line_end, bottomLeft, topLeft));

                            assert(point_final.size() <= 2);

                            if (point_final.size() == 0) continue;
                            if (point_final.size() == 2) linesSeg.push_back({point_final[0], point_final[1]});
                            else if (isCordInQuad(line_start, i, j)) linesSeg.push_back({line_start, point_final[0]});
                            else linesSeg.push_back({line_end, point_final[0]});
                        }
                    }
                }
                lineSegThisRow.push_back(linesSeg);
            }
            line_segements.push_back(lineSegThisRow);
        }
    }



public:
    GlobalWarp(vector<pair<int, int>> mesh, Mat img, int meshRow, int meshCol) {
        this->vertexes = mesh;
        this->img = img;
        this->meshRow = meshRow;
        this->meshCol = meshCol;

        find_lines();
        get_line_segment();

//        Scalar lineColor(0, 255, 0);
//        Scalar strat(255, 255, 0);
//        Scalar linecolor(255, 0, 255);
//        for (int i = 0; i < meshRow; i ++) {
//            for (int j = 0; j < meshCol; j ++) {
//                int index = i * (meshCol + 1) + j;
//                line(img, Point(vertexes[index].second, vertexes[index].first),
//                     Point(vertexes[index + 1].second, vertexes[index + 1].first),
//                     lineColor, 2);
//                line(img, Point(vertexes[index].second, vertexes[index].first),
//                     Point(vertexes[index + meshCol + 1].second, vertexes[index + meshCol + 1].first),
//                     lineColor, 2);
//            }
//        }
//        for (int i = 0; i < meshRow; i ++) {
//            for (int j = 0; j < meshCol; j ++) {
//                for (auto line: line_segements[i][j]) {
//                    cv::line(img, Point(line.first.second, line.first.first), Point(line.second.second, line.second.first),
//                             strat, 2);
//                }
//            }
//        }
//
//        for (auto line: lines) {
//            cv::line(img, Point(line.first.second, line.first.first), Point(line.second.second, line.second.first),
//                     linecolor, 1);
//        }
//
//        imshow("img", img);
//        waitKey(0);

    }

};

#endif //RECWARP_GLOBALWARP_H
