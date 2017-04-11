#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	NormalizeLayerTest()
: blob_bottom_(new Blob<Dtype>()),
  blob_top_(new Blob<Dtype>())
{
		Caffe::set_random_seed(1701);
		FillerParameter filler_param;
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
}
	virtual ~NormalizeLayerTest() { delete blob_bottom_; delete blob_top_; }

	void TestEqualValues() {
		int num = 1;
		int channels = 1;
		int height = 3;

		Blob<Dtype>* blob_vec = this->blob_bottom_vec_[0];
		std::vector<int> new_shape;
		new_shape.push_back(num);
		new_shape.push_back(channels);
		new_shape.push_back(height);

		blob_vec->Reshape(new_shape);
		Dtype* bottom_data = blob_vec->mutable_cpu_data();
		bottom_data[0] = 1;
		bottom_data[1] = 1;
		bottom_data[2] = 1;

		LayerParameter layer_param;
		NormalizeLayer<Dtype> layer(layer_param);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		//    // Now, check values
		const Dtype* top_data = this->blob_top_->cpu_data();
		Dtype first = top_data[0];
		Dtype second = top_data[1];
		Dtype third = top_data[2];

		const Dtype min_precision = 1e-5;
		const Dtype expected_first = 0.577350269;
		const Dtype expected_second = 0.577350269;
		const Dtype expected_third = 0.577350269;
		Dtype precision = std::max(Dtype(std::abs(expected_first * Dtype(1e-4))), min_precision);
		EXPECT_NEAR(first, expected_first, precision);
		EXPECT_NEAR(second, expected_second, precision);
		EXPECT_NEAR(third, expected_third, precision);
	}

	void TestUnchanged() {
		int num = 1;
		int channels = 1;
		int height = 3;

		Blob<Dtype>* blob_vec = this->blob_bottom_vec_[0];
		std::vector<int> new_shape;
		new_shape.push_back(num);
		new_shape.push_back(channels);
		new_shape.push_back(height);

		blob_vec->Reshape(new_shape);
		Dtype* bottom_data = blob_vec->mutable_cpu_data();
		bottom_data[0] = 0.0;
		bottom_data[1] = 1.0;
		bottom_data[2] = 0.0;

		LayerParameter layer_param;
		NormalizeLayer<Dtype> layer(layer_param);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		//    // Now, check values
		const Dtype* top_data = this->blob_top_->cpu_data();
		Dtype first = top_data[0];
		Dtype second = top_data[1];
		Dtype third = top_data[2];

		const Dtype min_precision = 1e-5;
		const Dtype expected_first = 0;
		const Dtype expected_second = 1.0;
		const Dtype expected_third = 0;
		Dtype precision_first_last = std::max(Dtype(std::abs(expected_first * Dtype(1e-4))), min_precision);
		Dtype precision_second = std::max(Dtype(std::abs(expected_first * Dtype(1e-4))), min_precision);
		EXPECT_NEAR(first, expected_first, precision_first_last);
		EXPECT_NEAR(second, expected_second, precision_second);
		EXPECT_NEAR(third, expected_third, precision_first_last);
	}

	void TestGradient() {
		int num = 1;
		int channels = 1;
		int height = 3;

		// Forward
		Blob<Dtype>* blob_bottom = this->blob_bottom_vec_[0];
		std::vector<int> new_shape;
		new_shape.push_back(num);
		new_shape.push_back(channels);
		new_shape.push_back(height);

		blob_bottom->Reshape(new_shape);
		Dtype* bottom_data = blob_bottom->mutable_cpu_data();
		bottom_data[0] = 1;
		bottom_data[1] = 0;
		bottom_data[2] = 0;

		LayerParameter layer_param;
		NormalizeLayer<Dtype> layer(layer_param);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		// Backward
		Blob<Dtype>* blob_top = this->blob_top_vec_[0];
		const Dtype* top_data = blob_top->cpu_data();
		Dtype first = top_data[0];
		Dtype second = top_data[1];
		Dtype third = top_data[2];

		cout << first << " - " << second << " - " << third << endl;

		Dtype* top_diff = blob_top->mutable_cpu_diff();
		top_diff[0] = -1.0;
		top_diff[1] =  5.0;
		top_diff[2] =  1.0;

		vector<bool> propagate_down;
		propagate_down.push_back(true);
		propagate_down.push_back(true);
		propagate_down.push_back(true);

		layer.Backward(this->blob_bottom_vec_, propagate_down, this->blob_top_vec_);

		const Dtype* bottom_diff = this->blob_bottom_->cpu_diff();
		Dtype g0 = bottom_diff[0];
		Dtype g1 = bottom_diff[1];
		Dtype g2 = bottom_diff[2];

		cout << g0 << " " << g1 << " " << g2 << endl;

	}

	Blob<Dtype>* const blob_bottom_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>* > blob_bottom_vec_;
	vector<Blob<Dtype>* > blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

//TYPED_TEST(NormalizeLayerTest, TestEqualValues) {
//	this->TestEqualValues();
//}
//
//TYPED_TEST(NormalizeLayerTest, TestUnchanged) {
//	this->TestUnchanged();
//}

TYPED_TEST(NormalizeLayerTest, TestGradient) {
	this->TestGradient();
}

}  // namespace caffe
