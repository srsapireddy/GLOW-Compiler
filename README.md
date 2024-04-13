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



