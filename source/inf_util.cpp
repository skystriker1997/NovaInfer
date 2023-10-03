#include "inf_util.hpp"

void set_multi_sink()
{
    using namespace spdlog::sinks;

    auto console_sink = std::make_shared<
            stdout_color_sink_mt>();
    console_sink->set_level(
            spdlog::level::warn);
    console_sink->set_pattern(
            "%H:%M:%S.%e %^%L%$ %v");

    auto file_sink =
            std::make_shared<basic_file_sink_mt>(
                    "./log/log");
    file_sink->set_level(
            spdlog::level::trace);
    file_sink->set_pattern(
            "%Y-%m-%d %H:%M:%S.%f %L %v");

    auto logger =
            std::shared_ptr<spdlog::logger>(
                    new spdlog::logger(
                            "multi_sink",
                            {console_sink, file_sink}));
    logger->set_level(
            spdlog::level::debug);
    spdlog::set_default_logger(
            logger);
}




Check::Check(): condition_(false) {};

Check& Check::operator()(bool condition) {
        condition_ = condition;
        return *this;
    }

void Check::operator << (std::string&& message) const {
        if(!condition_) {
            spdlog::critical(message);
            std::terminate();
        }
    }



