#include <chrono>
#include <cstdio>

#include "guided_filter.h"
#include "fast_guided_filter.h"

#ifndef NO_AUTO_SCHEDULE
#include "guided_filter_auto_schedule.h"
#include "fast_guided_filter_auto_schedule.h"
#endif

#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Usage: ./process guided.png input.png raduis eps s output.png output_fast.png\n"
               "e.g.: ./process guided.png input.png 8 0.0004 4 output.png output_fast.png\n");
        return 1;
    }

    Buffer<uint8_t> guided = load_image(argv[1]);
    Buffer<uint8_t> input = load_image(argv[2]);

    float eps = atof(argv[4]);
    int radius = atof(argv[3]);
    int s = atof(argv[5]);
    Buffer<uint8_t> output(input.width(), input.height(), 3);
    Buffer<uint8_t> output_fast(input.width(), input.height(), 3);

    guided_filter(guided, input, eps, output);

    // Manually-tuned version
    double best_manual_1 = benchmark([&]() {
        guided_filter(guided, input, eps, output);
        output.device_sync();
    });
    double best_manual_2 = benchmark([&]() {
        fast_guided_filter(guided, input, eps,radius,s, output_fast);
        output_fast.device_sync();
    });
    printf("Manually-tuned time(guided filter): %gms\n", best_manual_1 * 1e3);
    printf("Manually-tuned time(fast guided filter): %gms\n", best_manual_2 * 1e3);

#ifndef NO_AUTO_SCHEDULE
    // Auto-scheduled version
    double best_auto_1 = benchmark([&]() {
        guided_filter_auto_schedule(guided, input, eps, output);
        output.device_sync();
    });
    double best_auto_2 = benchmark([&]() {
        fast_guided_filter_auto_schedule(guided, input, eps, radius, s, output_fast);
        output_fast.device_sync();
    });
    printf("Auto-scheduled time(guided filter): %gms\n", best_auto_1 * 1e3);
    printf("Auto-scheduled time(fast guided filter): %gms\n", best_auto_2 * 1e3);
#endif

    convert_and_save_image(output, argv[6]);
    convert_and_save_image(output_fast, argv[7]);

    printf("Success!\n");
    return 0;
}