/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "atb_speed/log/log_sink_file.h"
#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <syscall.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include "atb_speed/utils/filesystem.h"

namespace atb_speed {
const int64_t MAX_LOG_FILE_SIZE = 1073741824; // 1G
const size_t MAX_LOG_FILE_COUNT = 5;

static bool GetLogToFileFlushEnv()
{
    const char *envLogToFileFlush = std::getenv("ATB_LOG_TO_FILE_FLUSH");
    return envLogToFileFlush != nullptr && strcmp(envLogToFileFlush, "1") == 0;
}

LogSinkFile::LogSinkFile(LogLevel level) : LogSink(level)
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream curTime;
    curTime << std::put_time(std::localtime(&tmpTime), "%Y%m%d%H%M%S");
    curTime_ = curTime.str();
    if (!FileSystem::Exists(fileDir_)) {
        FileSystem::Makedirs(fileDir_, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    std::stringstream filePath;
    filePath << fileDir_ << std::string("atb_speed_") << std::to_string(syscall(SYS_gettid)) << "_" << curTime_ << 
        "_" << fileCount_ << ".log";
    fileHandle_.open(filePath.str(), std::ios_base::out);
}

LogSinkFile::~LogSinkFile()
{
    fileHandle_.close();
    fileHandle_.clear();
}

void LogSinkFile::LogImpl(const LogEntity &logEntity)
{
    const int microsecond = 1000000;
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity.time);
    int us = 
        std::chrono::duration_cast<std::chrono::microseconds>(logEntity.time.time_since_epoch()).count() % microsecond;
    std::stringstream content;
    content << "[" << std::put_time(std::localtime(&tmpTime), "%F %T") << "." << us << "] [" <<
        LogLevelToString(logEntity.level) << "] [" << logEntity.processId << "] [" << logEntity.threadId << "] [" <<
        logEntity.fileName << ":" << logEntity.line << "]" << logEntity.content << std::endl;

    fileHandle_ << content.str();
    if (isFlush_) {
        fileHandle_.flush();
    }
    int64_t fileSize = static_cast<int64_t>(fileHandle_.tellp());
    if (fileSize >= MAX_LOG_FILE_SIZE) {
        fileHandle_.close();
        fileCount_++;
        if (fileCount_ == MAX_LOG_FILE_COUNT) {
            fileCount_ = 0;
        }
        std::stringstream filePath;
        filePath << fileDir_ << std::string("atb_speed_") << std::to_string(syscall(SYS_gettid)) << "_" << curTime_ << 
            "_" <<fileCount_ << ".log";
        fileHandle_.open(filePath.str(), std::ios_base::out);
    }
}
} // namespace AsdOps