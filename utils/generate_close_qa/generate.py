import argparse
from ast import literal_eval

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from tqdm import tqdm
import json


token = ''  # your access token to Llama2

model_id = 'meta-llama/Llama-2-13b-chat-hf'
batch_size = 4

# fp16
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', token=token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
    token=token
)
# fix a bug: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id

    
class QADataset(torch.utils.data.Dataset):
    def __init__(self, qa_annotations, start, end):
        self.qa_annotations = qa_annotations
        if start is not None and end is not None:
            self.qa_annotations = self.qa_annotations[start:end]

        print('preparing LM prompts...')
        self.lm_inputs = []
        for data in tqdm(self.qa_annotations):
            self.lm_inputs.append(self._get_lm_input(data['question'], data['answer']))

    def _get_lm_input(self, question, answer):
        return f"""<s>[INST] <<SYS>>
I'll provide a question and its correct answer. Generate three plausible, but incorrect, answers that closely resemble the correct one Make it challenging to identify the right answer. 
<</SYS>>

No preamble, get right to the three wrong answers and present them in a list format. Question: How many frying pans can i see on the shelf? Correct Answer: two pieces. Wrong Answers: [/INST] [\"one piece\", \"three piece\", \"five pieces\"] </s>
<s>[INST] No preamble, get right to the three wrong answers and present them in a list format. Question: What colour bowl did i carry from the plate stand? Correct Answer: green. Wrong Answers: [/INST] [\"blue\", \"black\", \"white\"] </s>
<s>[INST] No preamble, get right to the three wrong answers and present them in a list format. Question: What did i pour in the bowl? Correct Answer: boiling water. Wrong Answers: [/INST] [\"hot oil\", \"steamed milk\", \"warm broth\"] </s>
<s>[INST] No preamble, get right to the three wrong answers and present them in a list format. Question: {question} Correct Answer: {answer}. Wrong Answers: [/INST]

"""

    def __len__(self):
        return len(self.qa_annotations)
    
    def __getitem__(self, idx):
        return self.lm_inputs[idx]
    
    def get_data(self, idx):
        return self.qa_annotations[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    with open('annotations.EgoTimeQA.json', 'r') as f:
        qa_annotations = json.load(f)

    dataset = QADataset(qa_annotations, args.start, args.end)

    errors = 0
    pbar = tqdm(total=len(dataset))
    res = []
    for idx, out in enumerate(pipeline(
        dataset,
        batch_size=64,
        do_sample=True,
        temperature=0.5,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        return_full_text=False,
    )):
        pbar.set_description('Errors: %d' % errors)
        pbar.update(1)
        gen_result = out[0]['generated_text']

        try:  # may not generate the desired format
            wrong_answers = literal_eval(gen_result)
            assert isinstance(wrong_answers, list) and len(wrong_answers) == 3
            data = dataset.get_data(idx)
            data['wrong_answers'] = wrong_answers
            res.append(data)
            success = True
        except:
            errors += 1
            print(gen_result)

    print(f'#{len(res)} / {len(dataset)} samples generated!')
    
    if args.start is not None and args.end is not None:
        with open(f'tmp/annotations.EgoTimeQA_{args.start}_{args.end}.json', 'w') as f:
            json.dump(res, f)
    else:
        with open('annotations.EgoTimeQA.json', 'w') as f:
            json.dump(res, f)
