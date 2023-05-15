//
// Created by 母国宏 on 2023/5/5.
//

#ifndef RECWARP_LOCALWARP_H
#define RECWARP_LOCALWARP_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

const int INF = 1e8;

enum Direction {
    LEFT = 0,
    RIGHT,
    UP,
    DOWN
};

/*
 * LocalWarp类，执行算法中的第一步"Local Warping via Seam Carving"
 * 功能：输入图像，输出经第一步处理后作用在原图像上的网格
 */
class LocalWarp {

private:
    Mat img;        // 待处理的图片
    Mat img_origin; // 原始图片
    Mat img_gray;   // 灰度图
    Mat energyMap;  // 能量图
    Mat displacement_vertical;   // 记录每个像素点的垂直移动
    Mat displacement_horizontal; // 记录每个像素点的水平移动
    Mat mask;       // 掩码
    Mat seam;
    vector<pair<int, int>> mesh;    // 网格（从左上到右下）



    // 根据img计算能量图，结果存储在energyMap中
    void calcEnergyMap() {
        Mat x_grad, y_grad, abs_x_grad, abs_y_grad; // 梯度
        Mat img_blur;

//         采用gaussian blur减少噪声的影响
//        GaussianBlur(img, img_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

        // 转换为灰度图
        cvtColor(img, img_gray, CV_BGR2GRAY);


//        // 使用Scharr计算梯度
//        Scharr(img_gray, x_grad, -1, 1, 0, 1, 0, BORDER_DEFAULT);
//        Scharr(img_gray, y_grad, -1, 0, 1, 1, 0, BORDER_DEFAULT);
//
//        // 转换为绝对值
//        convertScaleAbs(x_grad, abs_x_grad);
//        convertScaleAbs(y_grad, abs_y_grad);
//
//        // 计算最终的能量图
//        Mat energyMat;
//        addWeighted(abs_x_grad, 0.5, abs_y_grad, 0.5, 1, energyMat);
//        energyMat.convertTo(energyMap, CV_32FC1);
//        energyMap /= 255;
//
//
        Mat m = Mat::zeros(img.rows, img.cols, CV_32FC1);

        // 计算forward energy
        for (int i = 1; i < img.rows; i ++) {
            for (int j = 0; j < img.cols; j ++) {
                int up = (i - 1) % img.rows;
                int left = (j - 1 + img.cols) % img.cols;
                int right = (j + 1) % img.cols;

                double mU = m.at<float>(up, j);
                double mL = m.at<float>(up, left);
                double mR = m.at<float>(up, right);

                double cU = abs(img_gray.at<uchar>(i, right) - img_gray.at<uchar>(i, left));
                double cL = abs(img_gray.at<uchar>(up, j) - img_gray.at<uchar>(i, left)) + cU;
                double cR = abs(img_gray.at<uchar>(up, j) - img_gray.at<uchar>(i, right)) + cU;

                // 计算最小的值
                double minValue = min(min(cU + mU, cL + mL), cR + mR);
                if (minValue == cU + mU) {
                    m.at<float>(i, j) = cU + mU;
                    energyMap.at<float>(i, j) = cU;
                }
                else if (minValue == cL + mL) {
                    m.at<float>(i, j) = cL + mL;
                    energyMap.at<float>(i, j) = cL;
                }
                else {
                    m.at<float>(i, j) = cR + mR;
                    energyMap.at<float>(i, j) = cR;
                }
            }
        }
        energyMap /= 255;

//        imshow("energyMap", energyMap);
//        waitKey(0);

    }


    // 找到最长的需填充边
    pair<int, int> find_longest(Direction &direction) {
        int maxLength, start, end;          // 最终结果
        int tmpLength, tmpStart, tmpEnd;   // 中间结果
        maxLength = start = end = tmpLength = tmpStart = tmpEnd = 0;

        bool isCounting = false;
        // LEFT
        for (int i = 0; i < img.rows; i ++) {
            // 为有效区域或到最后一个像素点
            uchar x = mask.at<uchar>(i, 0);
            if (mask.at<uchar>(i, 0) == 0 || i == img.rows - 1) {
                // 结束计数并比较
                if (isCounting) {
                    // 最后一个像素为无效点
                    if (mask.at<uchar>(i, 0) != 0) {
                        tmpLength ++;
                        tmpEnd ++;
                    }
                    if (tmpLength > maxLength) {
                        maxLength = tmpLength;
                        start = tmpStart;
                        end = tmpEnd;
                        direction = LEFT;
                    }
                }
                isCounting = false;
                tmpStart = tmpEnd = i + 1;
                tmpLength = 0;
            }
            // 无效区域
            else {
                tmpLength ++;
                tmpEnd ++;
                isCounting = true;
            }
        }

        // RIGHT
        tmpLength = tmpStart = tmpEnd = 0;
        isCounting = false;
        for (int i = 0; i < img.rows; i ++) {
            // 为有效区域或到最后一个像素点
            if (mask.at<uchar>(i, img.cols - 1) == 0 || i == img.rows - 1) {
                // 结束计数并比较
                if (isCounting) {
                    // 最后一个像素为无效点
                    if (mask.at<uchar>(i, img.cols - 1) != 0) {
                        tmpLength ++;
                        tmpEnd ++;
                    }
                    if (tmpLength > maxLength) {
                        maxLength = tmpLength;
                        start = tmpStart;
                        end = tmpEnd;
                        direction = RIGHT;
                    }
                }
                isCounting = false;
                tmpStart = tmpEnd = i + 1;
                tmpLength = 0;
            }
                // 无效区域
            else {
                tmpLength ++;
                tmpEnd ++;
                isCounting = true;
            }
        }

        // UP
        tmpLength = tmpStart = tmpEnd = 0;
        isCounting = false;
        for (int i = 0; i < img.cols; i ++) {
            // 为有效区域或到最后一个像素点
            if (mask.at<uchar>(0, i) == 0 || i == img.cols - 1) {
                // 结束计数并比较
                if (isCounting) {
                    // 最后一个像素为无效点
                    if (mask.at<uchar>(0, i) != 0) {
                        tmpLength ++;
                        tmpEnd ++;
                    }
                    if (tmpLength > maxLength) {
                        maxLength = tmpLength;
                        start = tmpStart;
                        end = tmpEnd;
                        direction = UP;
                    }
                }
                isCounting = false;
                tmpStart = tmpEnd = i + 1;
                tmpLength = 0;
            }
                // 无效区域
            else {
                tmpLength ++;
                tmpEnd ++;
                isCounting = true;
            }
        }

        // DOWN
        tmpLength = tmpStart = tmpEnd = 0;
        isCounting = false;
        for (int i = 0; i < img.cols; i ++) {
            // 为有效区域或到最后一个像素点
            if (mask.at<uchar>(img.rows - 1, i) == 0 || i == img.cols - 1) {
                // 结束计数并比较
                if (isCounting) {
                    // 最后一个像素为无效点
                    if (mask.at<uchar>(img.rows - 1, i) != 0) {
                        tmpLength ++;
                        tmpEnd ++;
                    }
                    if (tmpLength > maxLength) {
                        maxLength = tmpLength;
                        start = tmpStart;
                        end = tmpEnd;
                        direction = DOWN;
                    }
                }
                isCounting = false;
                tmpStart = tmpEnd = i + 1;
                tmpLength = 0;
            }
                // 无效区域
            else {
                tmpLength ++;
                tmpEnd ++;
                isCounting = true;
            }
        }

        return make_pair(start, end - 1);
    }


    // 利用动态规划计算seam
    vector<pair<int, int>> get_seam(Mat srcMap, Mat srcMask, Direction direction, int length) {
        Mat map = Mat::zeros(srcMap.rows, srcMap.cols, CV_32F);
        vector<pair<int, int>> pixels(length, {0, 0});
        int cnt = 0;    // pixel下标

        // 为无效区域赋无穷
        for (int i = 0; i < srcMap.rows; i ++) {
            for (int j = 0; j < srcMap.cols; j ++) {
                if (srcMask.at<uchar>(i, j) == 255) srcMap.at<float>(i, j) = INF;
            }
        }

        float minValue;
        int index;

        // 找到值最小的点
        switch (direction) {
            case LEFT:
            case RIGHT:
                // 动态规划
                for (int i = 1; i < srcMap.rows; i ++) {
                    for (int j = 0; j < srcMap.cols; j ++) {
                        if (j == 0) {
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(map.at<float>(i - 1, j),
                                                                               map.at<float>(i - 1, j + 1));
                        }
                        else if (j == srcMap.cols - 1) {
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(map.at<float>(i - 1, j - 1),
                                                                               map.at<float>(i - 1, j));
                        }
                        else {
                            float tmp = min(map.at<float>(i - 1, j - 1), map.at<float>(i - 1, j));
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(tmp, map.at<float>(i - 1, j + 1));
                        }
                    }
                }
                // 在最后一行找
                minValue = INF;
                index = 0;
                for (int i = 0; i < map.cols; i ++) {
                    if (map.at<float>(map.rows - 1, i) < minValue) {
                        minValue = map.at<float>(map.rows - 1, i);
                        index = i;
                    }
                }
//                pixels.push_back(make_pair(map.rows - 1, index));
                pixels[cnt].first = map.rows - 1;
                pixels[cnt].second = index;
                cnt ++;
                // 反向递推
                for (int i = map.rows - 2; i >= 0; i --) {
                    if (index == 0) {
                        if (map.at<float>(i, index) < map.at<float>(i, index + 1)) {
//                            pixels.push_back(make_pair(i, index));
                            pixels[cnt].first = i;
                            pixels[cnt].second = index;
                            cnt ++;
                        }
                        else {
//                            pixels.push_back(make_pair(i, index + 1));
                            pixels[cnt].first = i;
                            pixels[cnt].second = index + 1;
                            cnt ++;
                            index = index + 1;
                        }
                    }
                    else if (index == map.cols - 1) {
                        if (map.at<float>(i, index) < map.at<float>(i, index - 1)) {
//                            pixels.push_back(make_pair(i, index));
                            pixels[cnt].first = i;
                            pixels[cnt].second = index;
                            cnt ++;
                        }
                        else {
//                            pixels.push_back(make_pair(i, index - 1));
                            pixels[cnt].first = i;
                            pixels[cnt].second = index - 1;
                            cnt ++;
                            index = index - 1;
                        }
                    }
                    else {
                        float tmp;
                        int tmpIndex;
                        if (map.at<float>(i, index) < map.at<float>(i, index - 1)) {
                            tmp = map.at<float>(i, index);
                            tmpIndex = index;
                        }
                        else {
                            tmp = map.at<float>(i, index - 1);
                            tmpIndex = index - 1;
                        }
                        if (map.at<float>(i, index + 1) < tmp) {
                            tmpIndex = index + 1;
                        }
//                        pixels.push_back(make_pair(i, tmpIndex));
                        pixels[cnt].first = i;
                        pixels[cnt].second = tmpIndex;
                        cnt ++;
                        index = tmpIndex;
                    }
                }
                break;
            case UP:
            case DOWN:
                // 动态规划
                for (int j = 1; j < srcMap.cols; j ++) {
                    for (int i = 0; i < srcMap.rows; i ++) {
                        if (i == 0) {
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(map.at<float>(i, j - 1),
                                                                               map.at<float>(i + 1, j - 1));
                        }
                        else if (i == srcMap.rows - 1) {
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(map.at<float>(i - 1, j - 1),
                                                                               map.at<float>(i, j - 1));
                        }
                        else {
                            float tmp = min(map.at<float>(i - 1, j - 1), map.at<float>(i, j - 1));
                            map.at<float>(i, j) = srcMap.at<float>(i, j) + min(tmp, map.at<float>(i + 1, j - 1));
                        }
                    }
                }
                // 在最后一列找
                minValue = INF;
                index = 0;
                for (int i = 0; i < map.rows; i ++) {
                    if (map.at<float>(i, map.cols - 1) < minValue) {
                        minValue = map.at<float>(i, map.cols - 1);
                        index = i;
                    }
                }
//                pixels.push_back(make_pair(index, map.cols - 1));
                pixels[cnt].first = index;
                pixels[cnt].second = map.cols - 1;
                cnt ++;
                // 反向递推
                for (int i = map.cols - 2; i >= 0; i --) {
                    if (index == 0) {
                        if (map.at<float>(index, i) < map.at<float>(index + 1, i)) {
//                            pixels.push_back(make_pair(index, i));
                            pixels[cnt].first = index;
                            pixels[cnt].second = i;
                            cnt ++;
                        }
                        else {
//                            pixels.push_back(make_pair(index + 1, i));
                            pixels[cnt].first = index + 1;
                            pixels[cnt].second = i;
                            cnt ++;
                            index = index + 1;
                        }
                    }
                    else if (index == map.rows - 1) {
                        if (map.at<float>(index, i) < map.at<float>(index - 1, i)) {
//                            pixels.push_back(make_pair(index, i));
                            pixels[cnt].first = index;
                            pixels[cnt].second = i;
                            cnt ++;
                        }
                        else {
//                            pixels.push_back(make_pair(index - 1, i));
                            pixels[cnt].first = index - 1;
                            pixels[cnt].second = i;
                            cnt ++;
                            index = index - 1;
                        }
                    }
                    else {
                        float tmp;
                        int tmpIndex;
                        if (map.at<float>(index, i) < map.at<float>(index - 1, i)) {
                            tmp = map.at<float>(index, i);
                            tmpIndex = index;
                        }
                        else {
                            tmp = map.at<float>(index - 1, i);
                            tmpIndex = index - 1;
                        }
                        if (map.at<float>(index + 1, i) < tmp) {
                            tmpIndex = index + 1;
                        }
//                        pixels.push_back(make_pair(tmpIndex, i));
                        pixels[cnt].first = tmpIndex;
                        pixels[cnt].second = i;
                        cnt ++;
                        index = tmpIndex;
                    }
                }
                break;
        }

        return pixels;
    }


    // seam carve
    void seamCarve() {
        Direction direction;    // 方向
        int start, end;

        Mat tmpImg = img.clone();
        // 直到没有像素点可填
        while (true) {
            pair<int, int> longest_line = find_longest(direction);
            start = longest_line.first;
            end = longest_line.second;

            if (start == end) break;

            Mat sub_map, sub_mask;
            switch (direction) {
                case LEFT:
                case RIGHT:
                    energyMap(Range(start, end + 1), Range(0, energyMap.cols)).copyTo(sub_map);
                    mask(Range(start, end + 1), Range(0, mask.cols)).copyTo(sub_mask);
                    break;
                case UP:
                case DOWN:
                    energyMap(Range(0, energyMap.rows), Range(start, end + 1)).copyTo(sub_map);
                    mask(Range(0, energyMap.rows), Range(start, end + 1)).copyTo(sub_mask);
                    break;
            }

            // 得到seam
            int length = end - start + 1;
            vector<pair<int, int>> pixels = get_seam(sub_map, sub_mask, direction, length);

            // 根据seam更新energyMap, mask, img以及displacement
            int row, col;
            for (auto pixel: pixels) {
                switch (direction) {
                    case LEFT:
                    case RIGHT:
                        row = pixel.first + start;
                        col = pixel.second;
                        break;
                    case UP:
                    case DOWN:
                        row = pixel.first;
                        col = pixel.second + start;
                        break;
                }
//                tmpImg.at<Vec3b>(row, col) = {51, 204, 51};
                switch (direction) {
                    case LEFT:
                        for (int j = 0; j < col; j ++) {
                            img.at<Vec3b>(row, j) = img.at<Vec3b>(row, j + 1);
//                            tmpImg.at<Vec3b>(row, j) = tmpImg.at<Vec3b>(row, j + 1);
                            mask.at<uchar>(row, j) = mask.at<uchar>(row, j + 1);
                            displacement_horizontal.at<int>(row, j) += -1;   // 表示向左一个单位
                        }
                        break;
                    case RIGHT:
                        for (int j = img.cols - 1; j > col; j --) {
                            img.at<Vec3b>(row, j) = img.at<Vec3b>(row, j - 1);
//                            tmpImg.at<Vec3b>(row, j) = tmpImg.at<Vec3b>(row, j - 1);
                            mask.at<uchar>(row, j) = mask.at<uchar>(row, j - 1);
                            displacement_horizontal.at<int>(row, j) += 1;   // 表示向右一个单位
                        }
                        break;
                    case UP:
                        for (int j = 0; j < row; j ++) {
                            img.at<Vec3b>(j, col) = img.at<Vec3b>(j + 1, col);
//                            tmpImg.at<Vec3b>(j, col) = tmpImg.at<Vec3b>(j + 1, col);
                            mask.at<uchar>(j, col) = mask.at<uchar>(j + 1, col);
                            displacement_vertical.at<int>(j, col) += -1;   // 表示向上一个单位
                        }
                        break;
                    case DOWN:
                        for (int j = img.rows - 1; j > row; j --) {
                            img.at<Vec3b>(j, col) = img.at<Vec3b>(j - 1, col);
//                            tmpImg.at<Vec3b>(j, col) = tmpImg.at<Vec3b>(j - 1, col);
                            mask.at<uchar>(j, col) = mask.at<uchar>(j - 1, col);
                            displacement_vertical.at<int>(j, col) += 1;   // 表示向下一个单位
                        }
                        break;
                }
                // 使用相反色替代原图seam部分
                Vec3b p = img.at<Vec3b>(row, col);
                img.at<Vec3b>(row, col) = {(unsigned char)(255 - p[0]), (unsigned char)(255 - p[1]), (unsigned char)(255 - p[2])};
            }
            calcEnergyMap();
//            imshow("img", tmpImg);
//            waitKey(10);
        }
    }



public:
    explicit LocalWarp(Mat img, Mat mask) {
        this->img = img.clone();
        this->mask = mask.clone();
        this->displacement_horizontal = Mat::zeros(img.rows, img.cols, CV_32SC1);
        this->displacement_vertical = Mat::zeros(img.rows, img.cols, CV_32SC1);
        this->img_origin = img.clone();
        this->energyMap = Mat::zeros(this->img.rows, this->img.cols, CV_32FC1);
        this->seam = Mat::zeros(img.size(), CV_8UC3);

        calcEnergyMap();
        seamCarve();
    }

    // 得到local warp之后的mesh
    vector<pair<int, int>> get_warp_mesh(int meshRow, int meshCol) {
        // 计算每个quad里有多少行和列像素
        float rowPerQuad = img.rows / meshRow;
        float colPerQuad = img.cols / meshCol;

        // 得到矩形上的网格
        vector<pair<int, int>> vertexes((meshRow + 1) * (meshCol + 1), {0, 0});
        for (int i = 0; i < meshRow + 1; i ++) {
            for (int j = 0; j < meshCol + 1; j ++) {
                // 顶点在边框上
                if (i == meshRow) {
                    vertexes[i * (meshCol + 1) + j] = make_pair(img.rows - 1, round(j * colPerQuad));
                    continue;
                }
                else if (j == meshCol) {
                    vertexes[i * (meshCol + 1) + j] = make_pair(round(i * rowPerQuad), img.cols - 1);
                    continue;
                }
                vertexes[i * (meshCol + 1) + j] = make_pair(round(i * rowPerQuad), round(j * colPerQuad));
            }
        }

        // 根据displacement local warp矩形
        for (auto &vertex: vertexes) {
            int row = vertex.first;
            int col = vertex.second;
            int d_vertical = displacement_vertical.at<int>(row, col);
            int d_horizontal = displacement_horizontal.at<int>(row, col);
            // 更新
            vertex.first = row - d_vertical;
            vertex.second = col - d_horizontal;
        }

//        Scalar lineColor(0, 255, 0);
//        for (int i = 0; i < meshRow; i ++) {
//            for (int j = 0; j < meshCol; j ++) {
//                int index = i * (meshCol + 1) + j;
//                line(img_origin, Point(vertexes[index].second, vertexes[index].first),
//                     Point(vertexes[index + 1].second, vertexes[index + 1].first),
//                     lineColor, 2);
//                line(img_origin, Point(vertexes[index].second, vertexes[index].first),
//                     Point(vertexes[index + meshCol + 1].second, vertexes[index + meshCol + 1].first),
//                     lineColor, 2);
//            }
//        }
//
//        imshow("img", img_origin);
//        waitKey(0);

        return vertexes;
    }
};


#endif //RECWARP_LOCALWARP_H