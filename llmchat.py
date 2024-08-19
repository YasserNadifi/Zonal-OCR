import requests

class llmchat:
    def __init__(self, api_url, api_key, model_id):
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.conversation_history = []

    def send_prompt(self, prompt):
        # Append the new prompt to the conversation history
        self.conversation_history.append(f"User: {prompt}")

        # Create the prompt with conversation history
        conversation_context = "\n".join(self.conversation_history)
        print
        payload = {
            'model': self.model_id,
            'prompt': conversation_context,
            'stream': False
        }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            self.conversation_history.append(f"Model: {response_text}")

            return response_text
        else:
            raise Exception(f'Failed to chat with model: {response.status_code} {response.text}')

