# About
Win32 app for modeling dynamic systems and using image recognition for detecting sync.

# How to use

## Ready-To-Use Builds
Download the latest release from [releases](https://github.com/icosane/vervain/releases).

## Building Locally/using your own libraries
You can also run it using the ```.py``` files. 
>
Use ```syncdetect_local``` for running it in your IDE of choice. It expects all the additional files in the same directory.
>
Use ```syncdetect_executable``` for building it using [pyinstaller](https://pyinstaller.org/en/stable/) or [auto-py-to-exe](https://pypi.org/project/auto-py-to-exe/). 
>
For ```auto-py-to-exe``` you can use my build template ```build_settings_git.json```.

# Requirements
```
darkdetect==0.7.1
matplotlib==3.8.1
numpy==1.26.1
Pillow==10.0.1
PyQt5==5.15.10
pyqtdarktheme==2.1.0
scipy==1.11.3
tensorflow==2.14.0
PyQt5-sip==12.13.0
```
