import ollama

def get_llama3_response(prompt):
    response = ollama.chat(
        model='calebe/therapist-llama3.1',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']
