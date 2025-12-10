kill -9 $(lsof -t -i :11434)
ollama serve
ollama pull llama3.2:1b
