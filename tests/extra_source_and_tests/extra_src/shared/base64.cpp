#include <base64.h>

const char PADDING_CHAR = '=';
const char* ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const uint8_t DECODED_ALPHABET[128]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,0,0,0,63,52,53,54,55,56,57,58,59,60,61,0,0,0,0,0,0,0,0,
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,0,0,0,0,0,0,26,27,28,29,
30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,0,0,0,0,0};

/**
 * Given a string, this function will encode it in 64b (with padding)
 */
std::string encodeBase64(const std::string& binaryText)
{
    std::string encoded((binaryText.size()/3 + (binaryText.size()%3 > 0)) << 2, PADDING_CHAR);

    const char* bytes = binaryText.data();
    union
    {
        uint32_t temp = 0;
        struct
        {
            uint32_t first : 6, second : 6, third : 6, fourth : 6;
        } tempBytes;
    };
    std::string::iterator currEncoding = encoded.begin();

    for(uint32_t i = 0, lim = binaryText.size() / 3; i < lim; ++i, bytes+=3)
    {
        temp = bytes[0] << 16 | bytes[1] << 8 | bytes[2];
        (*currEncoding++) = ALPHABET[tempBytes.fourth];
        (*currEncoding++) = ALPHABET[tempBytes.third];
        (*currEncoding++) = ALPHABET[tempBytes.second];
        (*currEncoding++) = ALPHABET[tempBytes.first];
    }

    switch(binaryText.size() % 3)
    {
    case 1:
        temp = bytes[0] << 16;
        (*currEncoding++) = ALPHABET[tempBytes.fourth];
        (*currEncoding++) = ALPHABET[tempBytes.third];
        break;
    case 2:
        temp = bytes[0] << 16 | bytes[1] << 8;
        (*currEncoding++) = ALPHABET[tempBytes.fourth];
        (*currEncoding++) = ALPHABET[tempBytes.third];
        (*currEncoding++) = ALPHABET[tempBytes.second];
        break;
    }

    return encoded;
}

/**
 * Given a 64b padding-encoded string, this function will decode it.
 */
std::string decodeBase64(const std::string& base64Text)
{
    if (base64Text.empty())
        return "";

    size_t padding = 0;
    if (base64Text.size() >= 2) {
        if (base64Text[base64Text.size() - 1] == PADDING_CHAR) padding++;
        if (base64Text[base64Text.size() - 2] == PADDING_CHAR) padding++;
    }

    size_t decodedSize = (base64Text.size() / 4) * 3 - padding;
    std::string decoded(decodedSize, '\0');

    uint32_t temp = 0;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(base64Text.data());
    size_t decodedIndex = 0;

    for (size_t i = 0; i < base64Text.size() / 4 - 1; ++i, bytes += 4)
    {
        temp = DECODED_ALPHABET[bytes[0]] << 18 | DECODED_ALPHABET[bytes[1]] << 12 | DECODED_ALPHABET[bytes[2]] << 6 | DECODED_ALPHABET[bytes[3]];
        decoded[decodedIndex++] = (temp >> 16) & 0xFF;
        decoded[decodedIndex++] = (temp >> 8) & 0xFF;
        decoded[decodedIndex++] = temp & 0xFF;
    }

    temp = DECODED_ALPHABET[bytes[0]] << 18 | DECODED_ALPHABET[bytes[1]] << 12 | DECODED_ALPHABET[bytes[2]] << 6 | DECODED_ALPHABET[bytes[3]];
    if (padding == 0)
    {
        decoded[decodedIndex++] = (temp >> 16) & 0xFF;
        decoded[decodedIndex++] = (temp >> 8) & 0xFF;
        decoded[decodedIndex++] = temp & 0xFF;
    }
    else if (padding == 1)
    {
        decoded[decodedIndex++] = (temp >> 16) & 0xFF;
        decoded[decodedIndex++] = (temp >> 8) & 0xFF;
    }
    else if (padding == 2)
    {
        decoded[decodedIndex++] = (temp >> 16) & 0xFF;
    }

    return decoded;
}