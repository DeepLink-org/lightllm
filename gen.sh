curl http://0.0.0.0:8000/generate     \
    -X POST                             \
    -d '{"inputs":"How are you? ","parameters":{"max_new_tokens":20, "frequency_penalty":1}}' \
    -H 'Content-Type: application/json'
