#include <iostream>
#include <string>

namespace {

std::string json_error(const std::string& error) {
  return std::string("{\"status\":\"error\",\"error\":\"") + error + "\"}";
}

std::string extract_command(const std::string& line) {
  const std::string key = "\"command\"";
  const auto key_pos = line.find(key);
  if (key_pos == std::string::npos) return "";
  const auto colon_pos = line.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return "";
  const auto q1 = line.find('"', colon_pos + 1);
  if (q1 == std::string::npos) return "";
  const auto q2 = line.find('"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return line.substr(q1 + 1, q2 - q1 - 1);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  // Native game-loop scaffold.
  // This loop intentionally stays alive and processes command frames until SHUTDOWN.
  bool running = true;
  std::cout << "{\"status\":\"ok\",\"message\":\"k3d_native_daemon_started\"}" << std::endl;

  std::string line;
  while (running && std::getline(std::cin, line)) {
    if (line.empty()) {
      std::cout << json_error("empty_command") << std::endl;
      continue;
    }
    const std::string cmd = extract_command(line);
    if (cmd.empty()) {
      std::cout << json_error("missing_command") << std::endl;
      continue;
    }

    if (cmd == "PING" || cmd == "STATUS") {
      std::cout << "{\"status\":\"ok\",\"mode\":\"native\",\"running\":true}" << std::endl;
      continue;
    }
    if (cmd == "SHUTDOWN") {
      std::cout << "{\"status\":\"ok\",\"message\":\"shutdown_requested\"}" << std::endl;
      running = false;
      continue;
    }

    // Full TRM/Galaxy/PTX dispatch is implemented in the Python bridge daemon today.
    // This native daemon is the lifecycle anchor for the game paradigm cutover.
    std::cout << json_error("not_implemented_native_dispatch") << std::endl;
  }

  return 0;
}
