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
python infer.py models/mit-b1/ logits_mit_b1.pt
```
