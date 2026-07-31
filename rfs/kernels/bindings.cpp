#include <torch/extension.h>

void adamw_launch(void* parameter, const void* gradient, float* exp_avg,
                  float* exp_avg_sq, int64_t size, int scalar_type,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, float bias_correction1,
                  float bias_correction2, void* stream);
void ema_launch(float* state, const float* sample, int64_t size, float beta, void* stream);
void affine_identity_launch(void* matrix, void* output, int64_t size,
                            int64_t width, int scalar_type, double diagonal,
                            double scale, void* stream);
void symmetrize_launch(void* matrix, void* output, int64_t size,
                       int64_t width, int scalar_type, void* stream);
void parameter_step_launch(void* parameter, const void* update, int64_t size,
                           int parameter_type, int update_type, float lr,
                           float weight_decay, void* stream);

static int scalar_code(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32: return 0;
    case torch::kFloat16: return 1;
    case torch::kBFloat16: return 2;
    default: TORCH_CHECK(false, "Supported dtypes are float32, float16, and bfloat16");
  }
}

static void* current_stream() {
  // Optimizer kernels execute on PyTorch's default device stream in this
  // single-GPU benchmark. A null HIP stream is that stream.
  return nullptr;
}

static void same_device(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "RFS native kernels require GPU tensors");
  TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same GPU");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Tensors must be contiguous");
}

void adamw_step(torch::Tensor parameter, torch::Tensor gradient,
                torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                double lr, double beta1, double beta2, double eps,
                double weight_decay, double bias_correction1,
                double bias_correction2) {
  same_device(parameter, gradient);
  same_device(parameter, exp_avg);
  same_device(parameter, exp_avg_sq);
  TORCH_CHECK(parameter.numel() == gradient.numel(), "Parameter/gradient size mismatch");
  TORCH_CHECK(exp_avg.scalar_type() == torch::kFloat32 &&
              exp_avg_sq.scalar_type() == torch::kFloat32,
              "Adam moments must be float32");
  TORCH_CHECK(parameter.scalar_type() == gradient.scalar_type(), "Dtype mismatch");
  adamw_launch(parameter.data_ptr(), gradient.data_ptr(), exp_avg.data_ptr<float>(),
               exp_avg_sq.data_ptr<float>(), parameter.numel(), scalar_code(parameter),
               lr, beta1, beta2, eps, weight_decay, bias_correction1,
               bias_correction2, current_stream());
}

void ema(torch::Tensor state, torch::Tensor sample, double beta) {
  same_device(state, sample);
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              sample.scalar_type() == torch::kFloat32,
              "EMA tensors must be float32");
  TORCH_CHECK(state.sizes() == sample.sizes(), "EMA shape mismatch");
  ema_launch(state.data_ptr<float>(), sample.data_ptr<float>(), state.numel(),
             beta, current_stream());
}

torch::Tensor affine_identity(torch::Tensor matrix, double diagonal, double scale) {
  TORCH_CHECK(matrix.is_cuda() && matrix.is_contiguous(), "Matrix must be contiguous on GPU");
  TORCH_CHECK(matrix.scalar_type() == torch::kFloat32 ||
              matrix.scalar_type() == torch::kFloat64,
              "Matrix must be float32 or float64");
  TORCH_CHECK(matrix.dim() >= 2 && matrix.size(-1) == matrix.size(-2),
              "Expected a batch of square matrices");
  auto output = torch::empty_like(matrix);
  int matrix_type = matrix.scalar_type() == torch::kFloat32 ? 0 : 1;
  affine_identity_launch(matrix.data_ptr(), output.data_ptr(), matrix.numel(),
                         matrix.size(-1), matrix_type, diagonal, scale,
                         current_stream());
  return output;
}

torch::Tensor symmetrize(torch::Tensor matrix) {
  TORCH_CHECK(matrix.is_cuda() && matrix.is_contiguous(), "Matrix must be contiguous on GPU");
  TORCH_CHECK(matrix.scalar_type() == torch::kFloat32 ||
              matrix.scalar_type() == torch::kFloat64,
              "Matrix must be float32 or float64");
  TORCH_CHECK(matrix.dim() >= 2 && matrix.size(-1) == matrix.size(-2),
              "Expected a batch of square matrices");
  auto output = torch::empty_like(matrix);
  int matrix_type = matrix.scalar_type() == torch::kFloat32 ? 0 : 1;
  symmetrize_launch(matrix.data_ptr(), output.data_ptr(), matrix.numel(),
                    matrix.size(-1), matrix_type, current_stream());
  return output;
}

void parameter_step(torch::Tensor parameter, torch::Tensor update,
                    double lr, double weight_decay) {
  same_device(parameter, update);
  TORCH_CHECK(parameter.numel() == update.numel(), "Parameter/update size mismatch");
  parameter_step_launch(parameter.data_ptr(), update.data_ptr(), parameter.numel(),
                        scalar_code(parameter), scalar_code(update), lr,
                        weight_decay, current_stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("adamw_step", &adamw_step, "Fused AdamW step (HIP)");
  module.def("ema", &ema, "Fused EMA update (HIP)");
  module.def("affine_identity", &affine_identity, "Affine matrix plus identity (HIP)");
  module.def("symmetrize", &symmetrize, "Batched symmetrization (HIP)");
  module.def("parameter_step", &parameter_step, "Fused decoupled parameter step (HIP)");
}
