#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

cv::Mat gammaCorrection(cv::Mat src, float gamma)
{
	int cn = src.channels();
	if (cn == 1)
	{
		cv::Mat srcf(src.rows, src.cols, CV_32F);
		src.convertTo(srcf, CV_32F, 1 / 255.0);
		
		cv::Mat dst(src.rows, src.cols, CV_32F);

		for (int i = 0; i < src.rows; i++){
			for (int j = 0; j < src.cols; j++)
			{
				dst.at<float>(i, j) = std::powf(srcf.at<float>(i,j), gamma);
			}
		}

		cv::Mat redst(src.rows, src.cols, src.type());
		dst.convertTo(redst, src.type(), 256, -0.5);

		return redst;
	}
	else
	{
		cv::Mat srcf(src.rows, src.cols, CV_32FC3);
		src.convertTo(srcf, CV_32FC3, 1 / 255.0);

		cv::Mat srcArr[3];
		cv::split(srcf, srcArr);

		cv::Mat dstArr[3];
		for (int k = 0; k < 3; k++)
		{
			cv::Mat tmp(src.rows, src.cols, CV_32F);
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					tmp.at<float>(i, j) = std::powf(srcArr[k].at<float>(i, j), gamma);
				}
			}
			dstArr[k] = tmp;
		}

		cv::Mat dst(src.rows, src.cols, CV_32FC3);
		cv::merge(dstArr, 3, dst);

		cv::Mat redst(src.rows, src.cols, src.type());
		dst.convertTo(redst, src.type(), 256, -0.5);

		return redst;
	}
}

int main(int argc, char** argv)
{
	cv::Mat src = cv::imread("0001.jpg");
	imshow("0001", src);
	
	cv::Mat dst = gammaCorrection(src, 0.5);

	imshow("d", dst);

	cv::waitKey(0);

	return 0;
}