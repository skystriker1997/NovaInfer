#ifndef NOVA_INFER_UTIL
#define NOVA_INFER_UTIL

#include <string_view>
#include <spdlog/spdlog.h>
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <stdexcept>



void set_multi_sink();



class Check {
private:
    bool condition_;
public:
    Check();

    Check& operator()(bool condition);

    void operator << (std::string_view message) const;

    ~Check() = default;

};







#endif