# huggingface-test
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -r requirements/main.txt
(.venv) $ pip install -r requirements/dev.txt
```

## 推論
```
python infer.py models/mit-b1/ outputs/logits_mit_b1.pt
```

## Visualize
```
PYTHONPATH=. python scripts/visualize.py outputs/logits_mit_b4.pt [OUTPUT_PNG_FILENAME]
```

## ONNXへの変換
```
PYTHONPATH=. python scripts/segformer_onnx_export.py
```
