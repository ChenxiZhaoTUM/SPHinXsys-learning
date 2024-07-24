#ifndef _BASE64_H_
#define _BASE64_H_
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
typedef unsigned char BYTE;

std::string encodeBase64(const std::string& binaryText);
std::string decodeBase64(const std::string& base64Text);
#endif 