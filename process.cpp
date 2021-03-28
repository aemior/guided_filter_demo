#include <chrono>
#include <cstdio>

#include "guided_filter.h"

#ifndef NO_AUTO_SCHEDULE
#include "guided_filter_auto_schedule.h"
#endif

#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: ./process guided.png input.png eps output.png\n"
               "@TODO:radius is compile before"
               "e.g.: ./process guided.png input.png 0.0004 output.png\n");
        return 1;
    }

    Buffer<uint8_t> guided = load_image(argv[1]);
    Buffer<uint8_t> input = load_image(argv[2]);

    float eps = atof(argv[2]);
    Buffer<uint8_t> output(input.width(), input.height(), 3);

    guided_filter(guided, input, eps, output);

    // Manually-tuned version
    double best_manual = benchmark([&]() {
        guided_filter(guided, input, eps, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gms\n", best_manual * 1e3);

#ifndef NO_AUTO_SCHEDULE
    // Auto-scheduled version
    double best_auto = benchmark([&]() {
        guided_filter_auto_schedule(guided, input, eps, output);
        output.device_sync();
    });
    printf("Auto-scheduled time: %gms\n", best_auto * 1e3);
#endif

    convert_and_save_image(output, argv[4]);

    printf("Success!\n");
    return 0;
}