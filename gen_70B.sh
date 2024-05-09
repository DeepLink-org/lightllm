curl http://0.0.0.0:8000/generate     \
    -X POST                             \
    -d '{"inputs":"What daily habits might improve mental health considering factors like sleep quality, social interaction, and exercise impact according to psychology studies done recently? What daily habits might improve mental health considering factors like sleep quality, social interaction, and exercise impact according to psychology studies done recently? Do you knonw?","parameters":{"max_new_tokens":40, "frequency_penalty":1}}' \
    -H 'Content-Type: application/json'
