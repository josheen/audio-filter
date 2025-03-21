#!/bin/bash
set -e  # Exit on error

# Default values
RUN_TESTS=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    -test)
      RUN_TESTS=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done
mkdir -p build
cd build
cmake ..
make
if [ "$RUN_TESTS" = true ]; then
  echo "Running tests with ctest..."
  ctest --output-on-failure
  cd ..
fi

echo "Done."
