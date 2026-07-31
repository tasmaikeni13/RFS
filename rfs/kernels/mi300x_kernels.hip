#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

namespace {
constexpr int kThreads = 256;

template <typename T> __device__ inline float to_float(T value) {
  return static_cast<float>(value);
}
template <> __device__ inline float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T> __device__ inline T from_float(float value) {
  return static_cast<T>(value);
}
template <> __device__ inline __half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename scalar_t>
__global__ void adamw_kernel(scalar_t* parameter, const scalar_t* gradient,
                             float* exp_avg, float* exp_avg_sq, int64_t size,
                             float lr, float beta1, float beta2, float eps,
                             float weight_decay, float inv_bc1, float inv_sqrt_bc2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const float g = to_float(gradient[index]);
  const float m = beta1 * exp_avg[index] + (1.0f - beta1) * g;
  const float v = beta2 * exp_avg_sq[index] + (1.0f - beta2) * g * g;
  exp_avg[index] = m;
  exp_avg_sq[index] = v;
  const float update = (m * inv_bc1) / (sqrtf(v) * inv_sqrt_bc2 + eps);
  float value = to_float(parameter[index]);
  value = value * (1.0f - lr * weight_decay) - lr * update;
  parameter[index] = from_float<scalar_t>(value);
}

__global__ void ema_kernel(float* state, const float* sample, int64_t size, float beta) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) state[index] = beta * state[index] + (1.0f - beta) * sample[index];
}

template <typename T>
__global__ void affine_identity_kernel(const T* matrix, T* output,
                                       int64_t size, int64_t width,
                                       double diagonal, double scale) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const int64_t local = index % (width * width);
  output[index] = static_cast<T>(scale * matrix[index] +
                  ((local / width) == (local % width) ? diagonal : 0.0));
}

template <typename T>
__global__ void symmetrize_kernel(const T* matrix, T* output,
                                  int64_t size, int64_t width) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const int64_t matrix_size = width * width;
  const int64_t batch_start = (index / matrix_size) * matrix_size;
  const int64_t local = index % matrix_size;
  const int64_t row = local / width;
  const int64_t column = local % width;
  output[index] = static_cast<T>(0.5 *
      (matrix[index] + matrix[batch_start + column * width + row]));
}

template <typename parameter_t, typename update_t>
__global__ void parameter_step_kernel(parameter_t* parameter, const update_t* update,
                                      int64_t size, float lr, float weight_decay) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) return;
  float value = to_float(parameter[index]);
  value = value * (1.0f - lr * weight_decay) - lr * to_float(update[index]);
  parameter[index] = from_float<parameter_t>(value);
}

template <typename T>
void launch_adam_t(void* parameter, const void* gradient, float* exp_avg,
                   float* exp_avg_sq, int64_t size, float lr, float beta1,
                   float beta2, float eps, float weight_decay, float bc1,
                   float bc2, hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(adamw_kernel<T>),
      dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
      static_cast<T*>(parameter), static_cast<const T*>(gradient), exp_avg,
      exp_avg_sq, size, lr, beta1, beta2, eps, weight_decay, 1.0f / bc1,
      1.0f / sqrtf(bc2));
}

template <typename P, typename U>
void launch_parameter_t(void* parameter, const void* update, int64_t size,
                        float lr, float weight_decay, hipStream_t stream) {
  hipLaunchKernelGGL((parameter_step_kernel<P, U>),
      dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
      static_cast<P*>(parameter), static_cast<const U*>(update), size, lr, weight_decay);
}
}  // namespace

void adamw_launch(void* parameter, const void* gradient, float* exp_avg,
                  float* exp_avg_sq, int64_t size, int scalar_type,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, float bias_correction1,
                  float bias_correction2, void* raw_stream) {
  auto stream = reinterpret_cast<hipStream_t>(raw_stream);
  if (scalar_type == 0) launch_adam_t<float>(parameter, gradient, exp_avg, exp_avg_sq,
      size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, stream);
  else if (scalar_type == 1) launch_adam_t<__half>(parameter, gradient, exp_avg, exp_avg_sq,
      size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, stream);
  else launch_adam_t<hip_bfloat16>(parameter, gradient, exp_avg, exp_avg_sq,
      size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, stream);
}

void ema_launch(float* state, const float* sample, int64_t size, float beta, void* raw_stream) {
  hipLaunchKernelGGL(ema_kernel, dim3((size + kThreads - 1) / kThreads),
      dim3(kThreads), 0, reinterpret_cast<hipStream_t>(raw_stream), state, sample, size, beta);
}

void affine_identity_launch(void* matrix, void* output, int64_t size,
                            int64_t width, int scalar_type, double diagonal,
                            double scale, void* raw_stream) {
  auto stream = reinterpret_cast<hipStream_t>(raw_stream);
  if (scalar_type == 0)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(affine_identity_kernel<float>),
        dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
        static_cast<float*>(matrix), static_cast<float*>(output), size, width, diagonal, scale);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(affine_identity_kernel<double>),
        dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
        static_cast<double*>(matrix), static_cast<double*>(output), size, width, diagonal, scale);
}

void symmetrize_launch(void* matrix, void* output, int64_t size,
                       int64_t width, int scalar_type, void* raw_stream) {
  auto stream = reinterpret_cast<hipStream_t>(raw_stream);
  if (scalar_type == 0)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(symmetrize_kernel<float>),
        dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
        static_cast<float*>(matrix), static_cast<float*>(output), size, width);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(symmetrize_kernel<double>),
        dim3((size + kThreads - 1) / kThreads), dim3(kThreads), 0, stream,
        static_cast<double*>(matrix), static_cast<double*>(output), size, width);
}

void parameter_step_launch(void* parameter, const void* update, int64_t size,
                           int parameter_type, int update_type, float lr,
                           float weight_decay, void* raw_stream) {
  auto stream = reinterpret_cast<hipStream_t>(raw_stream);
#define DISPATCH_UPDATE(P) \
  if (update_type == 0) launch_parameter_t<P, float>(parameter, update, size, lr, weight_decay, stream); \
  else if (update_type == 1) launch_parameter_t<P, __half>(parameter, update, size, lr, weight_decay, stream); \
  else launch_parameter_t<P, hip_bfloat16>(parameter, update, size, lr, weight_decay, stream)
  if (parameter_type == 0) { DISPATCH_UPDATE(float); }
  else if (parameter_type == 1) { DISPATCH_UPDATE(__half); }
  else { DISPATCH_UPDATE(hip_bfloat16); }
#undef DISPATCH_UPDATE
}
