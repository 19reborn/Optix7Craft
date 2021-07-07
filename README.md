# Ray-Tracing-Project
Based on NVIDIA OptiX7.3 Frameworks.
## Building under Windows

- Install Required Packages
	- CUDA 10.1 or 11.3
	- OptiX 7.3 SDK
	- latest driver
	- cmake
- Download or clone the source repository

```bash
git clone https://github.com/19reborn/Ray-Tracing-Project.git
```

### Visual Studio

- Open `CMake GUI` from your start menu
	- point "source directory" to the downloaded source directory (not directory 'src')
	- point "build directory" to <source directory>/build (agree to create this directory when prompted)
	- click 'configure', then specify the generator as Visual Studio 2017 or 2019, and the Optional platform as x64. If CUDA, SDK, and compiler are all properly installed this should enable the 'generate' button. If not, make sure all dependencies are properly installed, "clear cache", and re-configure.
	- click 'generate' (this creates a Visual Studio project and solutions)
	- click 'open project' (this should open the project in Visual Studio)

### CLion

- Open folder as a CLion project
- add cmake option `-G "Visual Studio 16 2019"`

## Game Control

- <kbd>Q</kbd><kbd>Esc</kbd> Quit the game
- <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> Move Forward / Left / Backward / Right
- <kbd>Left Shift</kbd> Run
- <kbd>Space</kbd> Jump
- <kbd>M</kbd> Fly Mode
- <kbd>Left Ctrl</kbd> Landing in Fly Mode
- <kbd>F10</kbd> <kbd>F11</kbd> Zoom the window size - / +

