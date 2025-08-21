# Setup PhysiCell on Linux

This document describes the PhysiCell installation on a Debian Linux distribution.
If you run on another flavor, please adjust accordingly.

We will install everything PhysiCell related under the ~/src folder.
If you prefer another folder name, please adjust the commands accordingly.


## &#x1F427; Operating system library dependencies

### &#x2728; Update the package manager:

```bash
sudo apt update
sudo apt upgrade
```

### &#x2728; Install GCC (required by PhysiCell):

```bash
sudo apt install build-essential
```

### &#x2728; Install Image Magick (required by PhysiCell):

ImageMagick is used for making jpeg and gif images from PhysiCell svg image output.

The PhysiCell `Makefile` is written for **ImageMagick >=  version 7**, which requires a magick command in front of each ImageMagick command (e.g. magick convert instead of convert).
Many Linux distributions ship with **ImageMagick version 6**.
This is why we might have to tewak a bit the installation.

```bash
sudo apt install imagemagick
```

Now try:
```bash
magick --version
```

1. If ok: you have >= version 7 installed. You are all set!
2. If you receive: Command 'magick' not found, try:
```bash
convert --version
```
3. If ok: you have version 6 or less installed.

**If and only if you have <= version 6 installed**, you can follow the instruction below to generate a magick command that simply passes everything to the next command. This will make the PhysiCell Makefile work for you too.

`cd` to the folder where you have your manual installed binaries. e.g. `~/.local/bin/`

Run this code line by line.

```bash
echo '$*' > magick
```
```bash
chmod 775 magick
```
```bash
which magick
```

### &#x2728; Install FFmpeg, Tar, Gzip, and Unzip (required by PhysiCell):

```bash
sudo apt install ffmpeg tar gzip unzip
```

### &#x2728; Install the Qt 5 library (required by PhysiCell-Studio):
```
sudo apt install qtbase5-dev
```

### &#x2728; Install Python (required by PhysiCell-Studio and PhysiCell Data Loader):

Python is most probably already installed, but pip might be missing (required by PhysiCell-Studio and pcdl).

```bash
sudo apt install python3-pip
```


## &#x1F427; Basic PhyiCell installation

### &#x2728; Install PhysiCell:

```bash
install='Y'
uart='None'
if [ -d ~/src/PhysiCell ]
then
    echo "WARNING : /home/$USER/src/PhysiCell already exists! do you wanna re-install? data will be lost! [Y,N]"
    read uart
else
    uart='Y'
fi
if [ $install == $uart ]
then
    mkdir -p ~/src
    cd ~/src
    curl -L https://github.com/MathCancer/PhysiCell/archive/refs/tags/$(curl https://raw.githubusercontent.com/MathCancer/PhysiCell/master/VERSION.txt).zip > download.zip
    unzip download.zip
    rm download.zip
    rm -fr PhysiCell
    mv PhysiCell-$(curl https://raw.githubusercontent.com/MathCancer/PhysiCell/master/VERSION.txt) PhysiCell
else
    echo 'installation terminated.'
fi
```

### &#x2728; Test the PhyiCell installation with the template sample project:

Run this code line by line.

```bash
make data-cleanup clean reset
```
```bash
cd ~/src/PhysiCell
```
```bash
make template
```
```bash
make -j8
```
```bash
./project
```
```bash
make jpeg
```
```bash
make gif
```
```bash
make movie
```


## &#x1F427; Essential installation

We will generate a python3 environment with the default python installation, where we will install all PhysiCell modelling related python libraries.
We will name this python3 environment pcvenv (PhysiCell virtual environment).

### &#x2728; Install PhysiCell-Studio:

```bash
install='Y'
uart='None'
if [ -d ~/src/PhysiCell-Studio ]
then
    echo "WARNING : /home/$USER/src/PhysiCell-Studio already exists! do you wanna re-install? data will be lost! [Y,N]"
    read uart
else
    uart='Y'
fi
if [ $install == $uart ]
then
    cd ~/src
    python3 -m venv pcvenv
    if ! grep -Fq 'alias pcvenv=' ~/.bash_aliases
    then
        echo "alias pcvenv=\"source /home/$USER/src/pcvenv/bin/activate\"" >> ~/.bash_aliases
    fi
    source /home/$USER/src/pcvenv/bin/activate
    curl -L https://github.com/PhysiCell-Tools/PhysiCell-Studio/archive/refs/tags/v$(curl https://raw.githubusercontent.com/PhysiCell-Tools/PhysiCell-Studio/refs/heads/main/VERSION.txt).zip > download.zip
    unzip download.zip
    rm download.zip
    rm -fr PhysiCell-Studio
    mv PhysiCell-Studio-$(curl https://raw.githubusercontent.com/PhysiCell-Tools/PhysiCell-Studio/refs/heads/main/VERSION.txt) PhysiCell-Studio
    pip3 install -r PhysiCell-Studio/requirements.txt
    cd ~/src/pcvenv/bin/
    echo "python3 /home/$USER/src/PhysiCell-Studio/bin/studio.py \$*" > pcstudio
    chmod 775 pcstudio
    cd ~/src
else
    echo 'installation terminated.'
fi
```

### &#x2728; Test the PhysiCell-Studio installation:

Run this code line by line.

```bash
cd ~/src/PhysiCell
```
```bash
pcvenv
```
```bash
pcstudio
```

### &#x2728; Official PhysiCell Studio manual:

+ https://github.com/PhysiCell-Tools/Studio-Guide/tree/main


## &#x1F427; Advanced installation

### &#x2728; Install PhysiCell Data Loader (pcdl) and iPython:

Run this code line by line.

```bash
pcvenv
```
```bash
pip3 install pcdl ipython
```

### &#x2728; Test the pcdl installation:

Fire up a python shell.

```bash
ipython
```
Inside the python shell write:

```python
import pcdl
print(pcdl.__version__)
exit()
```

### &#x2728; Official pcdl manual:

+ https://github.com/PhysiCell-Tools/python-loader


## &#x1F427; IDE VSCode integration (optional)

1. Install vs code, either from your operating system’s app store or from https://code.visualstudio.com/ .

2. Generate a vs code profile for physicell:

```
File | New Window with Profile
Name: physicell
Icon: choose something cool.
Create
Add Folder: /home/<username>/src
click the profile icon (default is a gearwheel) on the left side bottom corner.
Profile > physicell
```

3. Open the Folder:

```
File | Open Folder… | /home/<username>/src | Open
Yes, I trust the authors
```

4. Install the official python and C++ extensions into the profile:

```
click the profile icon (default is a gearwheel) on the left side bottom corner.
Profile > physicell
Extension: Python Install
Extension: C/C++ Install
```

5. Link pcvenv (the python environment we generated above):

```
View | Command Palette… | Python: Select Interpreter | Enter interpreter path… | Find… | /home/<username>/src/pcvenv/bin/activate
```
