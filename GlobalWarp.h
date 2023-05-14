//
// Created by 母国宏 on 2023/5/8.
//

#ifndef RECWARP_GLOBALWARP_H
#define RECWARP_GLOBALWARP_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "lsd/lsd.h"


using namespace std;
using namespace cv;
using namespace Eigen;

typedef pair<double, double> PDD;
typedef pair<PDD, PDD> Line;

const double PI = acos(-1);

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
    vector<double> rotation;            // 旋转角度
    vector<vector<vector<int>>> line_to_rotation;// 直线与bin的对应关系
    double lambdaL;                     // L
    double lambdaB;                     // B
    vector<pair<int, int>> tmp_vertexes;// 中途更新顶点
    vector<vector<vector<MatrixXd>>> line_weight;


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

        if (fabs(p1_1.second - p1_2.second) < 0.1)
            k = 0, b = (p1_1.second + p1_2.second) / 2;
        if (fabs(p2_1.second - p2_2.second) < 0.1)
            k1 = 0, b1 = (p2_1.second + p2_2.second) / 2;

        if (fabs(p1_1.first - p1_2.first) < 0.1) {
            x = (p1_1.first + p1_2.first) / 2;
            y = k1 * x + b1;
        }
        else if (fabs(p2_1.first - p2_2.first) < 0.1) {
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


    // 计算shape能量矩阵内的单元
    MatrixXd get_shape_matrix_element(int index) {
        auto Aq = MatrixXd(8, 4);
        // 获取Aq
        pair<int, int> topLeft = vertexes[index];
        pair<int, int> topRight = vertexes[index + 1];
        pair<int, int> bottomLeft = vertexes[index + (meshCol + 1)];
        pair<int, int> bottomRight = vertexes[index + (meshCol + 1) + 1];
        Aq << topLeft.first, -topLeft.second, 1, 0,
              topLeft.second, topLeft.first, 0, 1,
              topRight.first, -topRight.second, 1, 0,
              topRight.second, topRight.first, 0, 1,
              bottomLeft.first, -bottomLeft.second, 1, 0,
              bottomLeft.second, bottomLeft.first, 0, 1,
              bottomRight.first, -bottomRight.second, 1, 0,
              bottomRight.second, bottomRight.first, 0, 1;
        MatrixXd A = Aq * (Aq.transpose() * Aq).inverse() * Aq.transpose() - MatrixXd::Identity(8, 8);
        return A;
    }


    // 计算shape能量函数矩阵
    // 尺寸：(8 * meshRow * meshCol) x (2 * vertexNum)
    SparseMatrix<double, RowMajor> get_shape_energy() {
        SparseMatrix<double, RowMajor> shape_matrix(8 * meshRow * meshCol, 2 * vertexes.size());
        // 得到矩阵单元后放入shape矩阵中的指定单元
        int quad_cnt = 0;
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                int index = i * (meshCol + 1) + j; // mesh序号
                MatrixXd A = get_shape_matrix_element(index);
                for (int m = 0; m < A.rows(); m ++) {
                    shape_matrix.insert(8 * quad_cnt + m, 2 * index) = A(m, 0);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * index + 1) = A(m, 1);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * index + 2) = A(m, 2);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * index + 3) = A(m, 3);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * (index + (meshCol + 1))) = A(m, 4);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * (index + (meshCol + 1)) + 1) = A(m, 5);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * (index + (meshCol + 1)) + 2) = A(m, 6);
                    shape_matrix.insert(8 * quad_cnt + m, 2 * (index + (meshCol + 1)) + 3) = A(m, 7);
                }
                quad_cnt ++;
            }
        }
        shape_matrix /= meshCol * meshRow;
        shape_matrix.makeCompressed();
        return shape_matrix;
    }


    // 分配直线到bin
    void init_line_to_rotation() {
        // 将直线分配到指定bin
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                line_weight[i][j] = vector<MatrixXd>(line_segements[i][j].size() * 2, MatrixXd(2, 8));
                if (line_segements[i][j].empty()) continue;

                vector<int> line_to_rotation_element = vector<int>(line_segements[i][j].size(), 0);
                for (int k = 0; k < line_segements[i][j].size(); k ++) {
                    Line line = line_segements[i][j][k];
                    PDD start = line.first;
                    PDD end = line.second;
                    double angle = atan((start.first - end.first) / (start.second - end.second));
                    int bin_id = round((angle + PI / 2) / (PI / 49));
                    assert(bin_id < 50);
                    line_to_rotation_element[k] = bin_id;
                }
                line_to_rotation[i][j] = line_to_rotation_element;
            }
        }
    }



    // 计算一个点与quad的双线性插值
    MatrixXd bilinear_interpolation(PDD p, int index) {
        PDD v1 = tmp_vertexes[index];
        PDD v2 = tmp_vertexes[index + 1];
        PDD v3 = tmp_vertexes[index + (meshCol + 1)];
        PDD v4 = tmp_vertexes[index + (meshCol + 1) + 1];

        auto v21 = PDD{v2.first - v1.first, v2.second - v1.second};
        auto v31 = PDD{v3.first - v1.first, v3.second - v1.second};
        auto v41 = PDD{v4.first - v1.first, v4.second - v1.second};
        auto p1 = PDD{p.first - v1.first, p.second - v1.second};

        auto a1 = v31.first;
        auto a2 = v21.first;
        auto a3 = v41.first - v21.first - v31.first;

        auto b1 = v31.second;
        auto b2 = v21.second;
        auto b3 = v41.second - v21.second - v31.second;

        auto px = p1.first;
        auto py = p1.second;

        double t1n, t2n;

        if (a3 == 0 && b3 == 0) {
            Matx21d t = Matx22d(v31.first, v21.first, v31.second, v21.second) * Matx21d{p1.first, p1.second};
            t1n = t(0, 0);
            t2n = t(1, 0);
        } else {
            auto a = (b2 * a3 - a2 * b3);
            auto b = (-a2 * b1 + b2 * a1 + px * b3 - a3 * py);
            auto c = px * b1 - py * a1;
            if (a == 0) {
                t2n = -c / b;
            } else {
                t2n = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
            }
            if (abs(a1 + t2n * a3) <= 1e-6) {
                t1n = (py - t2n * b2) / (b1 + t2n * b3);
            } else {
                t1n = (px - t2n * a2) / (a1 + t2n * a3);
            }
        }

//        auto m1 = v1 + t1n * (v3 - v1);
//        auto m4 = v2 + t1n * (v4 - v2);

        auto v1w = 1 - t1n - t2n + t1n * t2n;
        auto v2w = t2n - t1n * t2n;
        auto v3w = t1n - t1n * t2n;
        auto v4w = t1n * t2n;

        MatrixXd res(2, 8);
        res << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
                0, v1w, 0, v2w, 0, v3w, 0, v4w;
        return res;
    }


    // 更新rotation angle
    void update_rotation() {
        // 清空rotation
        for (int i = 0; i < rotation.size(); i ++) {
            rotation[i] = 0;
        }

        vector<int> bin_cnt = vector<int>(50, 0);
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                int index = i * (meshCol + 1) + j;
                PDD topLeft = tmp_vertexes[index];
                PDD topRight = tmp_vertexes[index + 1];
                PDD bottomLeft = tmp_vertexes[index + (meshCol + 1)];
                PDD bottomRight = tmp_vertexes[index + (meshCol + 1) + 1];
                MatrixXd meshMat(8, 1);
                meshMat << topLeft.first, topLeft.second, topRight.first, topRight.second,
                           bottomLeft.first, bottomLeft.second, bottomRight.first, bottomRight.second;
                if (line_to_rotation[i][j].empty()) continue;
                for (int k = 0; k < line_to_rotation[i][j].size(); k ++) {
                    Line line = line_segements[i][j][k];
                    PDD line_start = line.first;
                    PDD line_end = line.second;
                    double e_hat_angle = atan((line_start.first - line_end.first) / (line_start.second - line_end.second));
                    MatrixXd start_weight = line_weight[i][j][2 * k];
                    MatrixXd end_weight = line_weight[i][j][2 * k + 1];
                    double e_angle = atan(((start_weight * meshMat)(0, 0) - (end_weight * meshMat)(0, 0))
                            / ((start_weight * meshMat)(1, 0) - (end_weight * meshMat)(1, 0)));
                    double delta = e_angle - e_hat_angle;
                    rotation[line_to_rotation[i][j][k]] += delta;
                    bin_cnt[line_to_rotation[i][j][k]] += 1;
                }
            }
        }
        // 计算平均值
        for (int i = 0; i < 50; i ++) {
            if (bin_cnt[i] == 0) rotation[i] = 0;
            else rotation[i] /= bin_cnt[i];
        }
    }


    // 计算line能量函数矩阵
    // 尺寸：(2 * lineNum) x (2 * vertexNum)
    SparseMatrix<double, RowMajor> get_line_energy() {
        int line_num = 0;
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                line_num += line_segements[i][j].size();
            }
        }

        SparseMatrix<double, RowMajor> line_energy(2 * line_num, 2 * vertexes.size());
        int line_cnt = 0;
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                int index = i * (meshCol + 1) + j;
                if (line_segements[i][j].empty()) continue;
                for (int k = 0; k < line_segements[i][j].size(); k ++) {
                    Line line = line_segements[i][j][k];
                    auto start_weight = bilinear_interpolation(line.first, index);
                    auto end_weight = bilinear_interpolation(line.second, index);
                    line_weight[i][j][2 * k] = start_weight;
                    line_weight[i][j][2 * k + 1] = end_weight;
                    double theta = rotation[line_to_rotation[i][j][k]];
                    MatrixXd R(2, 2);
                    R << cos(theta), -sin(theta), sin(theta), cos(theta);
                    MatrixXd e_hat(2, 1);
                    e_hat << line.first.first - line.second.first, line.second.first - line.second.second;
                    MatrixXd e = start_weight - end_weight;
                    MatrixXd C = R * e_hat * (e_hat.transpose() * e_hat).inverse() * e_hat.transpose() * R.transpose()
                            - MatrixXd::Identity(2, 2);
                    MatrixXd Ce = C * e;
                    Ce = Ce * lambdaL;
                    for (int m = 0; m < Ce.cols(); m ++) {
                        if (m < 4) {
                            line_energy.insert(2 * line_cnt, 2 * index + m) = Ce(0, m);
                            line_energy.insert(2 * line_cnt + 1, 2 * index + m) = Ce(1, m);
                        }
                        else {
                            line_energy.insert(2 * line_cnt, 2 * (index + (meshCol + 1)) + m - 4) = Ce(0, m);
                            line_energy.insert(2 * line_cnt + 1, 2 * (index + (meshCol + 1)) + m - 4) = Ce(1, m);
                        }
                    }
                    line_cnt ++;
                }
            }
        }
        line_energy /= line_num;
        line_energy.makeCompressed();
        return line_energy;
    }


    // boundary energy
    SparseMatrix<double, RowMajor> get_boundary_energy(SparseVector<double> &b, int offset) {
        int n = 2 * (meshRow + 1) * (meshCol + 1);
        int xn = meshCol + 1;

        SparseMatrix<double, RowMajor> boundary_energy(n, n);
        for (int k = 0; k < n; k += 2 * xn) { // left boundary `x`
            boundary_energy.insert(k, k) = 1;
//            b.insert(k + offset) = 1;
        }

        for (int k = 2 * xn - 2; k < n; k += 2 * xn) { // right boundary `x`
            boundary_energy.insert(k, k) = 1;
            b.insert(k + offset) = img.cols - 1;
        }

        for (int k = 1; k < 2 * xn; k += 2) { // top boundary `y`
            boundary_energy.insert(k, k) = 1;
//            b.insert(k + offset) = 1;
        }

        for (int k = n - 2 * xn + 1; k < n; k += 2) { // bottom boundary `y`
            boundary_energy.insert(k, k) = 1;
            b.insert(k + offset) = img.rows - 1;
        }

        boundary_energy *= lambdaB;
        b *= lambdaB;
        boundary_energy.makeCompressed();
        return boundary_energy;
    }


    // global warp过程
    void globalWarp() {
        // shape energy
        auto shape_energy = get_shape_energy();
        for (int i = 0; i < 10; i ++) {
            cout << "iteration: " << i << "\t";
            auto line_energy = get_line_energy();
            SparseVector<double> b(shape_energy.rows() + line_energy.rows() + 2 * (meshRow + 1) * (meshCol + 1), 1);
            auto boundary_energy = get_boundary_energy(b, shape_energy.rows() + line_energy.rows());
            SparseMatrix<double, RowMajor> K(shape_energy.rows() + line_energy.rows() + boundary_energy.rows(), 2 * vertexes.size());
            K.topRows(shape_energy.rows()) = shape_energy;
            K.middleRows(shape_energy.rows() + 1, line_energy.rows()) = line_energy;
            K.bottomRows(boundary_energy.rows()) = boundary_energy;
            K.makeCompressed();

            // 更新V
            auto *solver = new SparseQR<SparseMatrix<double, RowMajor>, COLAMDOrdering<int>>();
//            auto *solver = new SimplicialCholesky<SparseMatrix<double>>(K);
            solver->compute(K);
            SparseMatrix<double> V = solver->solve(b);
            for (int j = 0; j < V.rows() - 1; j += 2) {
                tmp_vertexes[j] = make_pair((int)V.coeff(j, 0), (int)V.coeff(j + 1, 0));
            }

            // 更新theta
            update_rotation();
            cout << "finish" << endl;
        }
    }


public:
    GlobalWarp(vector<pair<int, int>> mesh, Mat img, int meshRow, int meshCol, double lambdaL, double lambdaB) {
        this->vertexes = mesh;
        this->img = img;
        this->meshRow = meshRow;
        this->meshCol = meshCol;
        this->rotation = vector<double>(50, 0);
        this->line_to_rotation = vector<vector<vector<int>>>(meshRow, vector<vector<int>>(meshCol, vector<int>()));
        this->lambdaL = lambdaL;
        this->lambdaB = lambdaB;
        this->tmp_vertexes = mesh;
        this->line_weight = vector<vector<vector<MatrixXd>>>(meshRow, vector<vector<MatrixXd>>(meshCol, vector<MatrixXd>()));

        find_lines();
        get_line_segment();
        init_line_to_rotation();
        globalWarp();


        Scalar lineColor(0, 255, 0);
        Scalar strat(255, 255, 0);
        Scalar linecolor(255, 0, 255);
        for (int i = 0; i < meshRow; i ++) {
            for (int j = 0; j < meshCol; j ++) {
                int index = i * (meshCol + 1) + j;
                line(img, Point(tmp_vertexes[index].second, tmp_vertexes[index].first),
                     Point(tmp_vertexes[index + 1].second, tmp_vertexes[index + 1].first),
                     lineColor, 2);
                line(img, Point(tmp_vertexes[index].second, tmp_vertexes[index].first),
                     Point(tmp_vertexes[index + meshCol + 1].second, tmp_vertexes[index + meshCol + 1].first),
                     lineColor, 2);
            }
        }

        imshow("img", img);
        waitKey(0);

    }

};

#endif //RECWARP_GLOBALWARP_H
