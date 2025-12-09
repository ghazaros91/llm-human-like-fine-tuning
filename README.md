# Setup & Run

## 0. Install and Run Ollama
```
brew install ollama
ollama serve
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

## 1. Create and activate a virtual environment (Python 3.12)

```
python3 -m venv venv
source venv/bin/activate
```


On Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\activate


```

## 2. Install dependencies

```
pip3 install -r requirements.txt
```

## 3. Run training

```
python3 main.py
```

## 4. Output

Fine-tuned HF model → outputs/

Optional GGUF export → outputs/model.gguf


---

If you want, I can fully regenerate the **complete README.md** including this sectio