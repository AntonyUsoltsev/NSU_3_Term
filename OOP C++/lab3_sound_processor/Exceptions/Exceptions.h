//
// Created by Antony on 01.12.2022.
//
#ifndef LAB3_SOUND_PROCESSOR_EXCEPTIONS_H
#define LAB3_SOUND_PROCESSOR_EXCEPTIONS_H

enum {
    BAD_PROG_ARGS_KEY = 2,
    MISSING_KEY = 3,
    BAD_PROG_ARGS_EXTENSION = 4,
    PROGRAM_STOPPED = 5,
    FILE_NOT_OPEN = 6,
    BAD_CONFIG_CONVERTER = 7,
    BAD_CONFIG_CONV_ARGS = 8,
    BAD_TIME = 9,
    BAD_FILE_PARAMS = 10
};

class Exceptions : std::exception {
private:
    std::string reason;
    int return_code;
public:
    explicit Exceptions(std::string reas, int code) : reason(std::move(reas)), return_code(code) {}

    std::string what_err() {
        return this->reason;
    }

    int ret_code() const {
        return this->return_code;
    }
};

#endif //LAB3_SOUND_PROCESSOR_EXCEPTIONS_H
