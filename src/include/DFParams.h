#ifndef DF_PARAM_H
#define DF_PARAM_H
#include <vector>
#include <string>
#include <map>

class DFParams {
public:
    std::vector<uint8_t> enc_;
    std::vector<uint8_t> erb_dec_;
    std::vector<uint8_t> df_dec_;
    std::map<std::string, std::map<std::string, std::string>> config_;

    DFParams(const std::string& enc_path, const std::string& erb_dec_path, const std::string& df_dec_path, const std::string& config_path);
    const std::map<std::string, std::string>& section(const std::string& section) const;

private:
    std::vector<uint8_t> read_file(const std::string& path);
    std::map<std::string, std::map<std::string, std::string>> parse_config(const std::string& path);
};
#endif
