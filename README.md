# Ray-Tracing-Project
Based on NVIDIA OptiX7.3 Frameworks.
## Building under Windows

- Install Required Packages
	- see above: CUDA 10.1, OptiX 7.3 SDK, latest driver, and cmake
- Download or clone the source repository
- Open `CMake GUI` from your start menu
	- point "source directory" to the downloaded source directory (not directory 'src')
	- point "build directory" to <source directory>/build (agree to create this directory when prompted)
	- click 'configure', then specify the generator as Visual Studio 2017 or 2019, and the Optional platform as x64. If CUDA, SDK, and compiler are all properly installed this should enable the 'generate' button. If not, make sure all dependencies are properly installed, "clear cache", and re-configure.
	- click 'generate' (this creates a Visual Studio project and solutions)
	- click 'open project' (this should open the project in Visual Studio)
