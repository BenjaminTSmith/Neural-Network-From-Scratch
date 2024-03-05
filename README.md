# Neural Network from Scratch
## Build
To build this project, create a build directory within the repo and run
```bash
cmake -S src -B build
```
Then, navigate to the build directory and run 
```bash
make
```
This will
create an executable binary for your system which you can then run.
(Make sure you have cmake and make installed).

## Dependencies
This projects requires the Eigen library to be included. Either place
the library in your usr/local/include or create an include directory and
edit the CMakeLists.txt.


