#include <cmath>
#include <iostream>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

int main(int argc, char *argv[]) {
  std::unique_ptr<tflite::Interpreter> interpreter;

  // Build model
  auto tfmodel = tflite::FlatBufferModel::BuildFromFile(argv[1]);

  if (tfmodel == nullptr) {
    std::cerr << "Failed to create tflite model." << std::endl;

    return 1;
  }

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*tfmodel, resolver);

  builder(&interpreter);

  if (interpreter == nullptr) {
    std::cerr << "Failed to create interpreter." << std::endl;

    return 1;
  }

  // Test loop
  for (uint64_t i = 0; i < 10; i++) {
    // Make input
    int32_t to = rand() % 128;
    int32_t *ptr = nullptr;
    const auto &inputs = interpreter->inputs();

    {
      TfLiteCustomAllocation allocation;

      // Aligned alloc
      ptr = reinterpret_cast<int32_t *>(aligned_alloc(64, sizeof(int32_t)));

      memcpy(ptr, &to, sizeof(int32_t));

      allocation.data = ptr;
      allocation.bytes = 1 * sizeof(int32_t);

      interpreter->ResizeInputTensor(inputs[0], {});
      interpreter->SetCustomAllocationForTensor(inputs[0], allocation);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cerr << "Set input failed." << std::endl;

      goto out;
    }

    // Inference
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Inference failed with error." << std::endl;

      goto out;
    }

    // Check output size
    if (interpreter->outputs().size() != 1) {
      std::cerr << "Unexpected output tensor size." << std::endl;

      goto out;
    }

    // Print output dimension
    {
      auto out = interpreter->output_tensor(0);

      std::cout << "Input: " << to << std::endl << "Output: [";
      for (auto iter : tflite::TfLiteIntArrayView(out->dims)) {
        std::cout << iter << " ";
      }
      std::cout << "]" << std::endl;
    }

    std::cout << "########## PRINT INTERPRETER STATE BEGIN ##########"
              << std::endl;

    tflite::PrintInterpreterState(interpreter.get());

    std::cout << "########### PRINT INTERPRETER STATE END ###########"
              << std::endl;

  out:
    // Cleanup
    free(ptr);
  }

  return 0;
}
