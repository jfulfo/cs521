import torch
import numpy as np
from itergen.main import IterGen

class SteeringHook:
    def __init__(self, model, layer_idx, direction, scale=2.0):
        self.model = model
        self.layer_idx = layer_idx
        self.direction = direction
        self.scale = scale
        self.handle = None
    
    def __enter__(self):
        def hook(module, input, output):
            h = output[0]
            delta = torch.tensor(self.direction * self.scale, device=h.device, dtype=h.dtype)
            return (h + delta.unsqueeze(0).unsqueeze(0),) + output[1:]
        self.handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)
        return self
    
    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()

def get_acts(model, tokenizer, text, layer, device='cuda'):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    acts = {}
    hook = model.model.layers[layer].register_forward_hook(lambda m,i,o: acts.update({'h': o[0].detach()}))
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    return acts['h'][0, -1, :].float().cpu().numpy()

def compute_steering_direction(model, tokenizer, layer, device='cuda'):
    valid = ["SELECT id FROM", "SELECT name FROM", "SELECT user_id FROM", "SELECT email FROM", "SELECT price FROM"]
    invalid = ["SELECT xyz FROM", "SELECT fake_col FROM", "SELECT nonexistent FROM", "SELECT asdf FROM", "SELECT wrong FROM"]
    
    valid_acts = np.array([get_acts(model, tokenizer, t, layer, device) for t in valid])
    invalid_acts = np.array([get_acts(model, tokenizer, t, layer, device) for t in invalid])
    
    d = valid_acts.mean(0) - invalid_acts.mean(0)
    return d / np.linalg.norm(d)

def exists_column(schema, col_name):
    return col_name.lower() in [c.lower() for c in schema.get('columns', [])]

def exists_table(schema, tbl_name):
    return tbl_name.lower() in [t.lower() for t in schema.get('tables', [])]

def parse_sql_schema(problem):
    schema = {'tables': [], 'columns': []}
    schema_text = problem.get('db_info', '')
    for line in schema_text.split('\n'):
        line = line.strip()
        if line.startswith('#') and '(' in line:
            parts = line[1:].strip().split('(')
            table = parts[0].strip()
            schema['tables'].append(table)
            if len(parts) > 1:
                cols = [c.strip() for c in parts[1].replace(')', '').split(',')]
                schema['columns'].extend(cols)
    return schema

def generate_sql_with_steering(iter_gen, problem, steer_dir, layer):
    iter_gen.start(problem['prompt'])
    with SteeringHook(iter_gen.model, layer, steer_dir, scale=2.0):
        out = iter_gen.forward()
    return iter_gen.structured_gen[0]

def generate_sql_baseline(iter_gen, problem):
    iter_gen.start(problem['prompt'])
    out = iter_gen.forward()
    return iter_gen.structured_gen[0]

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-0.5B"
    
    iter_gen_steer = IterGen(grammar='sql', model_id=MODEL_ID, recurrence_penalty=1.0, parse_output_only=True)
    
    layer = len(iter_gen_steer.model.model.layers) // 2
    steer_dir = compute_steering_direction(iter_gen_steer.model, iter_gen_steer.tokenizer, layer)
    print(f"computed steering direction at layer {layer}")
    
    problem = {
        'prompt': 'db_id: concert_singer\ndb_info: # stadium ( stadium_id , location , name , capacity , highest , lowest , average )\n# singer ( singer_id , name , country , song_name , song_release_year , age , is_male )\n# concert ( concert_id , concert_name , theme , stadium_id , year )\n# singer_in_concert ( concert_id , singer_id )\n\nquestion: What are the names of all stadiums?\nSQL:',
        'db_info': '# stadium ( stadium_id , location , name , capacity , highest , lowest , average )\n# singer ( singer_id , name , country , song_name , song_release_year , age , is_male )\n# concert ( concert_id , concert_name , theme , stadium_id , year )\n# singer_in_concert ( concert_id , singer_id )'
    }
    
    print("\nsteering:")
    out_steer = generate_sql_with_steering(iter_gen_steer, problem, steer_dir, layer)
    print(out_steer)

    del iter_gen_steer
    
    iter_gen_base = IterGen(grammar='sql', model_id=MODEL_ID, recurrence_penalty=0.7, parse_output_only=True)
    print("\nbaseline with recurrence penalty:")
    out_base = generate_sql_baseline(iter_gen_base, problem)
    print(out_base)