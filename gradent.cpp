/*利用[-1,0,1]求彩色图像的梯度图
  图像边界扩充采用反射扩充方式
*/

#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

//利用[-1,0,1]计算三通道图像的梯度
cv::Mat calculateGradentX(const cv::Mat& src)
{
	cv::Mat dx = cv::Mat::zeros(src.rows, src.cols, CV_32FC3);

	float *datasrc;
	float *datadx;
	for (int i = 0; i < src.rows; i++)
	{
		datasrc = (float*)(src.data + src.step*i);
		datadx = (float*)(dx.data + dx.step*i);
		for (int j = 1; j < src.cols - 1; j++)
		{
			datadx[j * 3] = datasrc[(j + 1) * 3] - datasrc[(j - 1 * 3)];
			datadx[j * 3 + 1] = datasrc[(j + 1) * 3 + 1] - datasrc[(j - 1) * 3 + 1];
			datadx[j * 3 + 2] = datasrc[(j + 1) * 3 + 2] - datasrc[(j - 1) * 3 + 2];
		}
	}
	return dx;
}

//利用[-1,0,1]T计算三通道图像的梯度
cv::Mat calculateGradentY(const cv::Mat& src)
{
	cv::Mat dy = cv::Mat::zeros(src.rows, src.cols, CV_32FC3);

	float *datasrc1, *datasrc3;
	float *datady;
	for (int i = 1; i < src.rows - 1; i++)
	{
		datasrc1 = (float*)(src.data + src.step*(i - 1));
		datasrc3 = (float*)(src.data + src.step*(i + 1));
		datady = (float*)(dy.data + dy.step*i);
		for (int j = 0; j < src.cols; j++)
		{
			datady[j * 3] = datasrc3[j * 3] - datasrc1[j * 3];
			datady[j * 3 + 1] = datasrc3[j * 3 + 1] - datasrc1[j * 3 + 1];
			datady[j * 3 + 2] = datasrc3[j * 3 + 2] - datasrc1[j * 3 + 2];
		}
	}
	return dy;
}

cv::Mat getGradentMap(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& maxDx, cv::Mat& maxDy)
{
	cv::Mat dst(dx.rows, dx.cols, CV_32F);

	float* datadx, *datady;
	for (int i = 0; i < dx.rows; i++)
	{
		datadx = (float*)(dx.data + dx.step*i);
		datady = (float*)(dy.data + dy.step*i);
		for (int j = 0; j < dx.cols; j++)
		{
			float magniteB = std::sqrtf(datadx[j * 3] * datadx[j * 3] + datady[j * 3] * datady[j * 3]);
			float magniteG = std::sqrtf(datadx[j * 3 + 1] * datadx[j * 3 + 1] + datady[j * 3 + 1] * datady[j * 3 + 1]);
			float magniteR = std::sqrtf(datadx[j * 3 + 2] * datadx[j * 3 + 2] + datady[j * 3 + 2] * datady[j * 3 + 2]);

			if (magniteB >= magniteG && magniteB >= magniteR)
			{
				dst.at<float>(i, j) = magniteB;
				maxDx.at<float>(i, j) = datadx[j * 3];
				maxDy.at<float>(i, j) = datady[j * 3];
			}
			else if (magniteG >= magniteB && magniteG >= magniteR)
			{
				dst.at<float>(i, j) = magniteG;
				maxDx.at<float>(i, j) = datadx[j * 3 + 1];
				maxDy.at<float>(i, j) = datady[j * 3 + 1];
			}
			else if (magniteR >= magniteB && magniteR >= magniteG)
			{
				dst.at<float>(i, j) = magniteR;
				maxDx.at<float>(i, j) = datadx[j * 3 + 2];
				maxDy.at<float>(i, j) = datady[j * 3 + 2];
			}
		}
	}
	return dst;
}

int main(int argc, char** argv)
{
	cv::Mat src = cv::imread("0001.jpg");
	imshow("src", src);

	cv::Mat srcf(src.rows, src.cols, CV_32FC3);
	src.convertTo(srcf, CV_32FC3);

	cv::Mat dx = calculateGradentX(srcf);
	cv::Mat dy = calculateGradentY(srcf);

	cv::Mat maxDx(dx.rows, dx.cols, CV_32F);
	cv::Mat maxDy(dy.rows, dy.cols, CV_32F);
	cv::Mat dst = getGradentMap(dx, dy, maxDx, maxDy);

	cv::waitKey(0);
	return 0;
}