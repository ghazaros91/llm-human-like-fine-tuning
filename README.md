# Setup & Run

## 1. Install Ollama
```
brew install ollama

## 2. Run Ollama

sh run_ollama.sh
```

## 3. Create and activate a virtual environment (Python 3.12)

```
python3 -m venv venv
source venv/bin/activate
```


On Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\activate


```

## 4. Install dependencies

```
pip3 install -r requirements.txt
```

## 5. Run training

```
python3 main.py
```

## 6. Output

Fine-tuned HF model → outputs/

Optional GGUF export → outputs/model.gguf


---

If you want, I can fully regenerate the **complete README.md** including this sectio