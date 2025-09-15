
from SPHINX import SPHINXModel

class VLAModel:
    
    def __init__(self, mode_path: str):
        self.model = SPHINXModel.from_pretrained(pretrained_path=mode_path, with_visual=True)

    def __call__(self, query, image) -> str:
        response = self.model.generate_response(query, image, max_gen_len=2048, temperature=0.1, top_p=0.75, seed=0)
        return response

