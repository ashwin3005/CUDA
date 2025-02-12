#include <time.h>

struct timer {
    struct timespec start_time, end_time;
};

void start_timer(    timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t -> start_time);
}

void stop_timer(struct timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t -> end_time);
}

double time_diff(struct timer * t) {
    double diff = (t -> end_time.tv_sec - t -> start_time.tv_sec) +
        (t -> end_time.tv_nsec - t -> start_time.tv_nsec) / 1000000000.0;
    return diff;
}
