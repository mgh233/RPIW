//
// Created by 母国宏 on 2023/5/6.
//

#ifndef RECWARP_TOOLS_H
#define RECWARP_TOOLS_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

/*
 * 工具类，提供一些需要使用的函数
 *
 * createMask: 生成图像的mask
 *
 */
class Tools {

private:
    // 填补mask中的空洞
    void fillHole(const Mat src, Mat &dst) {
        Size s_size = src.size();
        Mat temp = Mat::zeros(s_size.height + 2, s_size.width + 2, src.type());
        src.copyTo(temp(Range(1, s_size.height + 1), Range(1, s_size.width + 1)));

        // 使用flood fill算法将mask的外部区域填补上
        floodFill(temp, Point(0, 0), Scalar(255));

        Mat cutImg;
        temp(Range(1, s_size.height + 1), Range(1, s_size.width + 1)).copyTo(cutImg);

        dst = src | (~cutImg);
    }



public:
    // 生成mask
    Mat createMask(Mat img) {
        Mat mask;
        cvtColor(img, img, CV_BGR2GRAY);

        uchar thr = 252;    // 背景的阈值
        Mat temp = Mat::zeros(img.size(), CV_8UC1);
        for (int i = 0; i < img.rows; i ++) {
            for (int j = 0; j < img.cols; j ++) {
                if (img.at<uchar>(i, j) < thr) {
                    temp.at<uchar>(i, j) = 255; //覆盖有效部分
                }
            }
        }

        fillHole(temp, mask);
        mask = ~mask;   // 有效部分变为0

        // 腐蚀膨胀操作
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        Mat dilate_out;
        dilate(mask, dilate_out, element);
        dilate(dilate_out, dilate_out, element);
        dilate(dilate_out, dilate_out, element);

        Mat eroded_out;
        erode(dilate_out, eroded_out, element);
        return eroded_out;
    }


    // 缩小图片到1M个pixels，返回scale
    double shrinkImage(Mat src, Mat &dst) {
        int pixel_numbers = src.rows * src.cols;
        double scale = sqrt(pixel_numbers / 1000000.0);
        resize(src, dst, Size(), 1 / scale, 1 / scale);
        return scale;
    }


    // 通过长宽比得到合适的mesh行和列数
    pair<int, int> get_mesh_size(Mat img) {
        double scale = (double) img.rows / img.cols;
        int col = sqrt(400 / scale);
        int row = scale * col;
        return {row, col};
    }

};

#endif //RECWARP_TOOLS_H
