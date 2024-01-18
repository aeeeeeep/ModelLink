/**
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
#include "atb_speed/utils/str_split.h"
#include <sstream>

namespace atb_speed {
void StrSplit(const std::string &text, const char delimiter, std::vector<std::string> &result)
{
    std::istringstream iss(text);
    std::string subStr;
    while (getline(iss, subStr, delimiter)) {
        result.push_back(subStr);
    }
}

std::string GetFuncNameAndNameSpace(const std::string &inputStr)
{
    int spaceInd = 0;
    int leftBracketInd = 0;
    std::string extractStr;
    for (int i = 0; i < inputStr.size(); i++) {
        if (inputStr.at(i) == ' ') {
            spaceInd = i;
        } else if (inputStr.at(i) == '(') {
            leftBracketInd = i;
            break;
        }
    }
    // opGraph.name 只支持数字、字母、下划线，且长度不超过 128
    if (spaceInd >= 0 && (leftBracketInd - spaceInd) > 0) {
        int len;
        if (leftBracketInd - (spaceInd + 1) > 128) {
            len = 128;
        } else {
            len = leftBracketInd - (spaceInd + 1);
        }
        extractStr = inputStr.substr(spaceInd + 1, len);
    } else {
        extractStr = inputStr;
    }

    for (char &i : extractStr) {
        if (!isalnum(i) && i != '_') {
            i = '_';
        }
    }
    return extractStr;
}

} // namespace atb_speed