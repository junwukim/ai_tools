#include <cstdint>

#include "fc3_runnable_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {

constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

int main() {
  const tflite::Model* model = tflite::GetModel(g_fc3_runnable_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Schema mismatch: model=%d runtime=%d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Failed to register FullyConnected op");
    return 1;
  }

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return 1;
  }

  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr || input->type != kTfLiteFloat32) {
    MicroPrintf("Unexpected input tensor");
    return 1;
  }

  input->data.f[0] = 1.0f;
  input->data.f[1] = -2.0f;
  input->data.f[2] = 0.5f;
  input->data.f[3] = 3.0f;

  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return 1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr || output->type != kTfLiteFloat32) {
    MicroPrintf("Unexpected output tensor");
    return 1;
  }

  MicroPrintf("output[0] = %.6f", static_cast<double>(output->data.f[0]));
  MicroPrintf("expected  = 2.467875");
  return 0;
}
