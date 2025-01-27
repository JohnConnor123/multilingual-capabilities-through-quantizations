## Quickstart
P.s. This code works only on Linux and tested only with python3.12 
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
3. Download MERA benchmark:
```
git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git
git pull --all --rebase --recurse-submodules
```
3. Download llama.cpp project:
```

```
4. 