from methods.mas_base import MAS

class CoT(MAS):
    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config, method_config_name=method_config_name)
    
    def inference(self, sample):
        
        prompt = sample["query"] + "\n\nLet's think step by step."
        
        response = self.call_llm(prompt=prompt)

        return {"response": response}