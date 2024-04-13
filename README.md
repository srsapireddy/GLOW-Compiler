![image](https://github.com/srsapireddy/GLOW-Compiler/assets/32967087/6edc1317-98f6-4d23-b51e-5d543694b166)

Glow is a machine learning compiler and execution engine for hardware
accelerators.  It is designed to be used as a backend for high-level machine
learning frameworks.  The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs. This
library is in active development. The project plan is described in the Github
issues section and in the
[Roadmap](https://github.com/pytorch/glow/wiki/Glow-Roadmap) wiki page.

## How does it work?

Glow lowers a traditional neural network dataflow graph into a two-phase
strongly-typed [intermediate representation (IR)](./docs/IR.md). The high-level
IR allows the optimizer to perform domain-specific optimizations. The
lower-level instruction-based address-only IR allows the compiler to perform
memory-related optimizations, such as instruction scheduling, static memory
allocation and copy elimination. At the lowest level, the optimizer performs
machine-specific code generation to take advantage of specialized hardware
features. Glow features a lowering phase which enables the compiler to support a
high number of input operators as well as a large number of hardware targets by
eliminating the need to implement all operators on all targets. The lowering
phase is designed to reduce the input space and allow new hardware backends to
focus on a small number of linear algebra primitives.
The design philosophy is described in an [arXiv paper](https://arxiv.org/abs/1805.00907).

![](./docs/3LevelIR.png)

## Installation 
Guide: https://github.com/pytorch/glow

### GLOW INSTALLATION STEPS On Ubuntu (16.04)

--------------------------------------------------------------------------------------------------------------------------------------------

### Glow Compiler Prerequisites:

Operating system: Ubuntu 16.04LTS

RAM: Minimum 16GB

SWAP MEMORY: Minimum 12GB to 20GB

Memory Needed: 70GB

Total Memory needed: Minimum 150GB(LLVM&GLOW)

------------------------------------------------------------------------------------------------------------------------------------------------

### Glow Compiler Dependencies:

LLVM 8.0.1

Clang 8.0.1

Anaconda 3

Pytorch, if GPU is used, needs to install CUDA 10.1 and cuDNN 7.1

------------------------------------------------------------------------------------------------------------------------------------------------

### Glow Compiler Process

```
Step 1: Download the glow repository from the GitHub

$git clone pytorch/glow

$cd glow

Step 2: Glow depends on a few submodules: google test, onyx, and a library for FP16 conversions. To get them from the glow directory, run the following:

$git submodule update --init --recursive

Step 3: If Protobuf is not installed install it by using the shell script, version should be 2.6.1, PATH: glow/utils/, run a shell script

$./install_protobuf.sh

Step 4: Create a build directory in glow

$mkdir build

$cd build # Change working directory to build

$cmake -DCMAKE_BUILD_TYPE=Release ../ # Now run cmake in Release mode, providing the Glow source directory as the path. This will build files into ......( It will take 4 to 8 hours or more based on RAM and SWAP memory), if cmake is not installed install it by running following command

$sudo apt install cmake

step 5: Run the make command to compile the source code

$make

Step 6: Run make install to install the library

$make install
```
-----------------------------------------------------------------------------------------------------------------------------------------------------

### Testing Glow:

A few test programs that use Glow's C++ API are found under the examples/ subdirectory. The mnist, cifar10, fr2en, and ptb programs train and run digit recognition, image classification, and language modeling benchmarks, respectively.
To run these programs, build Glow in Release mode, then run the following commands to download the cifar10, mnist, and ptb databases.

$python ../glow/utils/download_datasets_and_models.py --all-datasets

Now, run the examples. Note that the databases should be in the current working directory.

$./bin/mnist

$./bin/cifar10

$./bin/fr2en

$./bin/ptb

$./bin/char-rnn

### If everything goes well you should see:

mnist: pictures from the mnist digits database

cifar10: image classifications that steadily improve

fr2en: an interactive French-to-English translator

ptb: decreasing perplexity on the dataset as the network trains

char-rnn: generates random text based on some document

## Commands
### Running example model 
```
./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU

./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU -dump-ir -dump-graph-DAG

./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU -dump-graph-DAG=DAG.dot

./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU -dump-dir=/root/dev/build_/bin/dumpDir

cd /root/dev/build_/bin/
./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU -dump-ir=1

./image-classifier /root/dev/build_/tests/images/mnist/*.png -image-mode=0to1 -m=/root/dev/mnist_model.onnx -model-input-name=input.1 -backend=CPU -dump-ir-after-all-passes -dump-ir-before-all-passes > ir_dump.txt
```
### Example Run
![WhatsApp Image 2024-04-09 at 04 36 43_0763df22](https://github.com/srsapireddy/GLOW-Compiler/assets/32967087/768d7e92-72bf-4d63-b36b-58a55d219c58)


### Directed acyclic graph (DAG) of the model 
```
dot -Tpng DAG.dot -o DAG.png
```
### DAG file contents
![WhatsApp Image 2024-04-09 at 04 45 29_59dc9a61](https://github.com/srsapireddy/GLOW-Compiler/assets/32967087/6934fba0-b5bb-4a8b-901c-029b2541aadb)



