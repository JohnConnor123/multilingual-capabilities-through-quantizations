## Quickstart
P.s. This code works only on Linux and tested only with python3.12 and cuda12.1
1. Create enviroment:
```
python -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ && python setup.py build && python setup.py install && cd ..
```
P.s. If you get the error:
```
libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```
when running the AutoGPTQ installation command, then just run the command `unset LD_LIBRARY_PATH` (more details: https://github.com/pytorch/pytorch/issues/111469#issue-1949414420)

P.s. If you get the error:
```
In file included from /usr/local/cuda-12.1/include/cuda_runtime.h:83,
                 from <command-line>:
/usr/local/cuda-12.1/include/crt/host_config.h:132:2: error: #error -- unsupported GNU version! gcc versions later than 12 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
  132 | #error -- unsupported GNU version! gcc versions later than 12 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
      |  ^~~~~
error: command '/usr/local/cuda-12.1/bin/nvcc' failed with exit code 1
```
when running the AutoGPTQ installation command, then install gcc/g++ v12 and set it as default compiler:
Installation gcc/g++ v12:
```
sudo apt install gcc-12 g++-12
```
Setting gcc/g++ v12 as default - paste in script.sh this:
```
cd /usr/bin
for f in gcc cpp g++ gcc-ar gcc-nm gcc-ranlib gcov gcov-dump gcov-tool lto-dump; do
    ln -vsf $f-12 $f
done
```
and run it via `sudo bash script.sh` (more details: https://askubuntu.com/a/1510476)

3. Download MERA benchmark:
```
git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git
git pull --all --rebase --recurse-submodules
cd MERA/lm-evaluation-harness
pip install -e .[vllm]
cd ../..
```
After that use mera evaluate commands in MERA directory

3. Download llama.cpp project:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cd ..
```
P.s. If you want to speed up the build process, add `-j <number of threads>`. For example, the command below will use 8 threads to build llama.cpp:
`cmake --build build --config Release -j 8`

4. 
