
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <chrono>
#include <cmath> 

struct TrainingXForm {
	Eigen::Matrix<float, 3, 4> start;
	Eigen::Matrix<float, 3, 4> end;
};

// Evaluates a polynomial (of degree DEGREE) given it's coefficients at a specific point)
// using numerically stable Horner-scheme (https://en.wikipedia.org/wiki/Horner%27s_method)
template<size_t DEGREE, typename Scalar>
Scalar evaluatePolynomialHornerScheme(Scalar x,
                                      std::array<Scalar, DEGREE+1> const& coefficients)
{
      auto ret = Scalar{0};
      for(auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
      {
          ret = ret * x + *it;
      }
      return ret;
}

Eigen::Matrix<float, 3, 4> get_xform_given_rolling_shutter(const TrainingXForm& training_xform, const Eigen::Vector4f& rolling_shutter, const Eigen::Vector2f& uv, float motionblur_time) {
	float pixel_t = rolling_shutter.x() + rolling_shutter.y() * uv.x() + rolling_shutter.z() * uv.y() + rolling_shutter.w() * motionblur_time;

	Eigen::Vector3f pos = training_xform.start.col(3) + (training_xform.end.col(3) - training_xform.start.col(3)) * pixel_t;

	Eigen::Quaternionf rot = Eigen::Quaternionf(training_xform.start.block<3, 3>(0, 0)).slerp(pixel_t, Eigen::Quaternionf(training_xform.end.block<3, 3>(0, 0)));


	Eigen::Matrix<float, 3, 4> rv;
	rv.col(3) = pos;
	rv.block<3, 3>(0, 0) = Eigen::Quaternionf(rot).normalized().toRotationMatrix();
	return rv;
}

Eigen::Vector3f f_theta_undistortion(const Eigen::Vector2f& uv, const float* params, const Eigen::Vector3f& error_direction) {

	float norm = sqrtf(uv.x()*uv.x() + uv.y()*uv.y());
	float alpha = params[0] + norm * (params[1] + norm * (params[2] + norm * (params[3] + norm * params[4])));
	float sin_alpha, cos_alpha;
	sincosf(alpha, &sin_alpha, &cos_alpha);
	if (cos_alpha <= std::numeric_limits<float>::min() || norm == 0.f) {
		return error_direction;
	}
	sin_alpha *= 1.f / norm;
	return { sin_alpha * uv.x(), sin_alpha * uv.y(), cos_alpha };
}

template <typename T>
void apply_camera_distortion(const T* extra_params, const T u, const T v, T* du, T* dv) {
	const T k1 = extra_params[0];
	const T k2 = extra_params[1];
	const T p1 = extra_params[2];
	const T p2 = extra_params[3];

	const T u2 = u * u;
	const T uv = u * v;
	const T v2 = v * v;
	const T r2 = u2 + v2;
	const T radial = k1 * r2 + k2 * r2 * r2;
	*du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
	*dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

template <typename T>
void iterative_camera_undistortion(const T* params, T* u, T* v) {
	// Parameters for Newton iteration using numerical differentiation with
	// central differences, 100 iterations should be enough even for complex
	// camera models with higher order terms.
	const uint32_t kNumIterations = 100;
	const float kMaxStepNorm = 1e-10f;
	const float kRelStepSize = 1e-6f;

	Eigen::Matrix2f J;
	const Eigen::Vector2f x0(*u, *v);
	Eigen::Vector2f x(*u, *v);
	Eigen::Vector2f dx;
	Eigen::Vector2f dx_0b;
	Eigen::Vector2f dx_0f;
	Eigen::Vector2f dx_1b;
	Eigen::Vector2f dx_1f;

	for (uint32_t i = 0; i < kNumIterations; ++i) {
		const float step0 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(0)));
		const float step1 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(1)));
		apply_camera_distortion(params, x(0), x(1), &dx(0), &dx(1));
		apply_camera_distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
		apply_camera_distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
		apply_camera_distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
		apply_camera_distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
		J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
		J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
		J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
		J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
		const Eigen::Vector2f step_x = J.inverse() * (x + dx - x0);
		x -= step_x;
		if (step_x.squaredNorm() < kMaxStepNorm) {
			break;
		}
	}

	*u = x(0);
	*v = x(1);
}


const char* initCamRays_doc = R"igl_Qu8mg5v7(
Based on the camera intrinsic parameters it initializes the 3D rays in camera space
)igl_Qu8mg5v7";
npe_function(_initCamRays)
npe_doc(initCamRays_doc)
npe_arg(cameraIntrinsic, dense_float)
npe_arg(cameraDistortion, npe_matches(cameraIntrinsic))
npe_arg(imgWidth, int)
npe_arg(imgHeight, int)
npe_arg(distortionMode, std::string)
npe_begin_code()
	// Initialize the output array
	npe_Matrix_cameraIntrinsic cameraRays(imgWidth * imgHeight, 5);
	cameraRays.setZero();
	

	float distortionParams[7] = {cameraDistortion(0,0), cameraDistortion(0,1), cameraDistortion(0,2), cameraDistortion(0,3),
								 cameraDistortion(0,4), cameraDistortion(0,5), cameraDistortion(0,6)};

	float intrinsicParams[4] = {cameraIntrinsic(0,0), cameraIntrinsic(0,1), cameraIntrinsic(0,2), cameraIntrinsic(0,3)};

	for (int py = 0; py < imgHeight; py++) 
	{
		for (int px = 0; px < imgWidth; px++)
		{
			Eigen::RowVector2f uv;
			Eigen::RowVector2f uv_normalized;
						
			if (distortionMode == "FTheta")
			{
				uv = {px, py};
				uv_normalized = {uv.x() / (float)imgWidth, uv.y() / (float)imgHeight};
				cameraRays.template block(py*imgWidth + px, 0, 1, 2) = uv_normalized;
				cameraRays.template block(py*imgWidth + px, 2, 1, 3) = f_theta_undistortion({uv.x() - intrinsicParams[0], uv.y() - intrinsicParams[1]}, distortionParams, {0.f, 0.f, 1.f}).transpose();
			}
			else
			{
				uv = {px + 0.5, py + 0.5};
				uv_normalized = {uv.x() / (float)imgWidth, uv.y() / (float)imgHeight};
				cameraRays.template block(py*imgWidth + px, 0, 1, 2) = uv_normalized;
				Eigen::RowVector3f dir = {(uv.x() - intrinsicParams[0]) / intrinsicParams[2], (uv.y() - intrinsicParams[1]) / intrinsicParams[3], 1.0f};

				if (distortionMode == "Iterative")
				{
					iterative_camera_undistortion(distortionParams, &dir.x(), &dir.y());
				}

				cameraRays.template block(py*imgWidth + px, 2, 1, 3) = dir;
			}
		}
	}

    return npe::move(cameraRays);

npe_end_code()


const char* cam2WorldRays_doc = R"igl_Qu8mg5v7(
Transforms the camera rays to world rays (also compensating for rolling shutter)
)igl_Qu8mg5v7";
npe_function(_cam2WorldRays)
npe_doc(cam2WorldRays_doc)
npe_arg(cameraRays, dense_float)
npe_arg(transformStart, npe_matches(cameraRays))
npe_arg(transformEnd, npe_matches(cameraRays))
npe_arg(rollingShutter, npe_matches(cameraRays))
npe_begin_code()

	// Initialize the output array
	npe_Matrix_cameraRays worldRays(cameraRays.rows(), 6);
	
	TrainingXForm training_xform;
	training_xform.start = transformStart; 
	training_xform.end = transformEnd;

	Eigen::Vector4f rollingShutterParams(Eigen::Map<Eigen::Vector4f>(rollingShutter.data(), rollingShutter.cols()*rollingShutter.rows()));

	for (int pixelInd = 0; pixelInd < cameraRays.rows(); pixelInd++)
	{
		Eigen::Matrix<float, 3, 4> xform = get_xform_given_rolling_shutter(training_xform, rollingShutterParams, {cameraRays(pixelInd,0), cameraRays(pixelInd,1)}, 0.f);
		worldRays.template block(pixelInd,0,1,3) = xform.col(3).transpose();
		worldRays.template block(pixelInd,3,1,3) = (xform.block<3, 3>(0, 0) * cameraRays.template block(pixelInd,2,1,3).transpose()).normalized().transpose();
	}

    return npe::move(worldRays);

npe_end_code()


const char* batchCam2WorldRays_doc = R"igl_Qu8mg5v7(
Transforms the camera rays to world rays (also compensating for rolling shutter)
)igl_Qu8mg5v7";
npe_function(_batchCam2WorldRays)
npe_doc(cam2WorldRays_doc)
npe_arg(cameraRays, dense_float)
npe_arg(transformMatrices, npe_matches(cameraRays))
npe_arg(rollingShutter, npe_matches(cameraRays))
npe_begin_code()

	// Initialize the output array
	npe_Matrix_cameraRays worldRays(cameraRays.rows(), 6);
	
	for (int rayInd = 0; rayInd < cameraRays.rows(); rayInd++)
	{
		TrainingXForm training_xform;
		training_xform.start = transformMatrices.template block(6*rayInd,0,3,4); 
		training_xform.end = transformMatrices.template block(6*rayInd+3,0,3,4);

		Eigen::Vector4f rollingShutterParams(Eigen::Map<Eigen::Vector4f>(rollingShutter.row(rayInd).data(), 4));
		
		Eigen::Matrix<float, 3, 4> xform = get_xform_given_rolling_shutter(training_xform, rollingShutterParams, {cameraRays(rayInd,0), cameraRays(rayInd,1)}, 0.f);
		worldRays.template block(rayInd,0,1,3) = xform.col(3).transpose();
		worldRays.template block(rayInd,3,1,3) = (xform.block<3, 3>(0, 0) * cameraRays.template block(rayInd,2,1,3).transpose()).normalized().transpose();

	}

    return npe::move(worldRays);

npe_end_code()




const char* batchPix2WorldRays_doc = R"igl_Qu8mg5v7(
Transforms a batch of pixels into a batch of world rays (also compensating for rolling shutter)
)igl_Qu8mg5v7";
npe_function(_batchPix2WorldRays)
npe_doc(batchPix2WorldRays_doc)
npe_arg(batch, dense_int)
npe_arg(cameraIntrinsic, dense_float)
npe_arg(cameraDistortion, dense_float)
npe_arg(rollingShutter, dense_float)
npe_arg(transformMatrices, dense_float)
npe_arg(distortionType, dense_int)

npe_begin_code()

	// Initialize the output array
	npe_Matrix_cameraIntrinsic worldRays(batch.rows(), 6);
	worldRays.setZero();

	for (int batchIdx = 0; batchIdx < batch.rows(); batchIdx++)
	{
		int camIdx = batch(batchIdx, 0);
		int frameIdx = batch(batchIdx, 1);

		float distortionParams[7] = {cameraDistortion(camIdx,0), cameraDistortion(camIdx,1), cameraDistortion(camIdx,2), cameraDistortion(camIdx,3),
									cameraDistortion(camIdx,4), cameraDistortion(camIdx,5), cameraDistortion(camIdx,6)};

		float intrinsicParams[4] = {cameraIntrinsic(camIdx,0), cameraIntrinsic(camIdx,1), cameraIntrinsic(camIdx,2), cameraIntrinsic(camIdx,3)};

		float imgWidth = cameraIntrinsic(camIdx, 4);
		float imgHeight = cameraIntrinsic(camIdx, 5);

		Eigen::Vector3f camRay;
		Eigen::Vector2f uv;
		Eigen::Vector2f uv_normalized;

		if (distortionType(camIdx, 0) == 0) // F-theta
		{
			uv = {batch(batchIdx,2), batch(batchIdx,3)}; // convention for NVIDIA F-theta model is that pixels are centered at integer values
			uv_normalized = {uv.x() / imgWidth, uv.y() / imgHeight};
			camRay = f_theta_undistortion({uv.x() - intrinsicParams[0], uv.y() - intrinsicParams[1]}, distortionParams, {0.f, 0.f, 1.f});
		}
		else  // pinhole
		{
			uv = {batch(batchIdx,2) + 0.5, batch(batchIdx,3) + 0.5};
			uv_normalized = {uv.x() / imgWidth, uv.y() / imgHeight};			
			camRay = {(uv.x() - intrinsicParams[0]) / intrinsicParams[2], (uv.y() - intrinsicParams[1]) / intrinsicParams[3], 1.0f};
			if (distortionType(camIdx, 0) == 2)
			{
				iterative_camera_undistortion(distortionParams, &camRay.x(), &camRay.y());
			}

		}

		// If camera has a global shutter all the coefficients in rollingShutter are zero and there is only
		// a single extrinsic matrix per frame available
		Eigen::Matrix<float, 3, 4> xform;
		if (rollingShutter.isZero())
		{
			xform =  transformMatrices.template block(3*frameIdx,0,3,4); 
		}
		else{
			TrainingXForm training_xform;
			training_xform.start = transformMatrices.template block(6*frameIdx,0,3,4); 
			training_xform.end = transformMatrices.template block(6*frameIdx+3,0,3,4);
			
			Eigen::Vector4f rollingShutterParams(Eigen::Map<Eigen::Vector4f>(rollingShutter.row(camIdx).data(), 4));
			
			xform = get_xform_given_rolling_shutter(training_xform, rollingShutterParams, {uv_normalized.x(), uv_normalized.y()}, 0.f);
		}

		worldRays.template block(batchIdx,0,1,3) = xform.col(3).transpose();
		worldRays.template block(batchIdx,3,1,3) = (xform.block<3, 3>(0, 0) * camRay).normalized().transpose();
	}

    return npe::move(worldRays);

npe_end_code()


const char* batchPix2CamRays_doc = R"igl_Qu8mg5v7(
Transforms a batch of pixels into a batch of camera rays
)igl_Qu8mg5v7";
npe_function(_batchPix2CamRays)
npe_doc(batchPix2CamRays_doc)
npe_arg(batch, dense_int)
npe_arg(cameraIntrinsic, dense_float)
npe_arg(cameraDistortion, dense_float)
npe_arg(rollingShutter, dense_float)
npe_arg(transformMatrices, dense_float)
npe_arg(distortionType, dense_int)

npe_begin_code()

	// Initialize the output array
	npe_Matrix_cameraIntrinsic camRays(batch.rows(), 3);
	camRays.setZero();

	for (int batchIdx = 0; batchIdx < batch.rows(); batchIdx++)
	{
		int camIdx = batch(batchIdx, 0);

		float distortionParams[7] = {cameraDistortion(camIdx,0), cameraDistortion(camIdx,1), cameraDistortion(camIdx,2), cameraDistortion(camIdx,3),
									cameraDistortion(camIdx,4), cameraDistortion(camIdx,5), cameraDistortion(camIdx,6)};

		float intrinsicParams[4] = {cameraIntrinsic(camIdx,0), cameraIntrinsic(camIdx,1), cameraIntrinsic(camIdx,2), cameraIntrinsic(camIdx,3)};

		float imgWidth = cameraIntrinsic(camIdx, 4);
		float imgHeight = cameraIntrinsic(camIdx, 5);

		Eigen::Vector3f camRay;
		Eigen::Vector2f uv;
		Eigen::Vector2f uv_normalized;

		if (distortionType(camIdx, 0) == 0)
		{
			uv = {batch(batchIdx,2), batch(batchIdx,3)};
			uv_normalized = {uv.x() / imgWidth, uv.y() / imgHeight};
			camRay = f_theta_undistortion({uv.x() - intrinsicParams[0], uv.y() - intrinsicParams[1]}, distortionParams, {0.f, 0.f, 1.f});
		}
		else
		{
			uv = {batch(batchIdx,2) + 0.5, batch(batchIdx,3) + 0.5};
			uv_normalized = {uv.x() / imgWidth, uv.y() / imgHeight};
			camRay = {(uv.x() - intrinsicParams[0]) / intrinsicParams[2], (uv.y() - intrinsicParams[1]) / intrinsicParams[3], 1.0f};
			if (distortionType(camIdx, 0) == 2)
			{
				iterative_camera_undistortion(distortionParams, &camRay.x(), &camRay.y());
			}
		}

		camRays(batchIdx,0) = camRay.x();
		camRays(batchIdx,1) = camRay.y();
		camRays(batchIdx,2) = camRay.z();
	}
    return npe::move(camRays);
npe_end_code()
