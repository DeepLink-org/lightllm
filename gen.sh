curl http://0.0.0.0:8000/generate     \
    -X POST                             \
    -d '{"inputs":"Hello","parameters":{"max_new_tokens":128, "frequency_penalty":1}}' \
    -H 'Content-Type: application/json'
