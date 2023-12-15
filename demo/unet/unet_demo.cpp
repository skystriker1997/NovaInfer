#include "graph.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

void PreprocessImage(const cv::Mat &image, nova_infer::Tensor<float> &output) {
    Check check;
    check(!image.empty()) << "failed to preprocess the image; the image is empty";
    check(output.Channels()==3) << "failed to preprocess the image; the output tensor should have 3 channels";
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(512, 512));
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32FC3);
    std::vector<cv::Mat> splitted_image(3);
    cv::split(rgb_image, splitted_image);
    Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp(512, 512);
    for(int i=0; i<3; i++) {
        cv::cv2eigen(splitted_image[i], tmp);
        output.WriteMatrix(i) = tmp;
        output.WriteMatrix(i) /= 255.f;
    }
}



int main(int argc, char* argv[]) {
    Check check;
    check(argc == 4) << "failed to execute the demo of unet; correct usage: ./unet_demo [image path] [pnnx_param path] [pnnx_bin path]\n";

    std::string image_path = argv[1];
    std::string param_path = argv[2];
    std::string bin_path = argv[3];

    cv::Mat image = cv::imread(image_path);

    check(!image.empty()) << "failed to execute the demo of unet; failed to load image";

    nova_infer::Batchf input;
    input.emplace_back(std::vector<int>{3, 512, 512});

    PreprocessImage(image, input[0]);

    nova_infer::Graph graph(param_path, bin_path);
    for(auto &tensor: input) {
        graph.AppendBatch(tensor);
    }

    graph.Forward();

    std::shared_ptr<nova_infer::Batchf> output = graph.GetOutput();

    for(nova_infer::Tensor<float> &tensor: *output) {
        auto mtx0 = tensor.ReadMatrix(0);
        auto mtx1 = tensor.ReadMatrix(1);
        check(mtx0.rows()==512 && mtx0.cols()==512 && mtx1.rows()==512 && mtx1.cols()==512) << "failed to export the black and white image; the output size is not 512*512";
        cv::Mat segmented_image(512, 512, CV_32F);
        for(int r=0; r<512; r++) {
            for(int c=0; c<512; c++) {
                segmented_image.at<float>(r,c) = mtx0(r,c)<mtx1(r,c)?255.f:0.f;
            }
        }
        cv::imwrite(cv::String("unet_output.jpg"), segmented_image);
    }
    return 0;
}


