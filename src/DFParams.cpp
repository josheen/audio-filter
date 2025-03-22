#include "DFParams.h"
#include <fstream>

DFParams::DFParams(const std::string& enc_path, const std::string& erb_dec_path, const std::string& df_dec_path, const std::string& config_path) {
    enc_ = read_file(enc_path);
    erb_dec_ = read_file(erb_dec_path);
    df_dec_ = read_file(df_dec_path);
    config_ = parse_config(config_path);
}

const std::map<std::string, std::string>& DFParams::section(const std::string& section) const {
    auto it = config_.find(section);
    if (it == config_.end()) {
        throw std::runtime_error("Config section not found: " + section);
    }
    return it->second;
}

std::vector<uint8_t> DFParams::read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::map<std::string, std::map<std::string, std::string>> DFParams::parse_config(const std::string& path) {
    std::map<std::string, std::map<std::string, std::string>> config;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::string line;
    std::string current_section;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines or comments
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;

        if (line.front() == '[' && line.back() == ']') {
            // Found section
            current_section = line.substr(1, line.size() - 2);
        } else {
            size_t delimiter = line.find('=');
            if (delimiter != std::string::npos) {
                std::string key = line.substr(0, delimiter);
                std::string value = line.substr(delimiter + 1);
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                config[current_section][key] = value;
            }
        }
    }
    return config;
}
