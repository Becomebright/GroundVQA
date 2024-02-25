import os
import argparse
from ast import literal_eval

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np
from tqdm import tqdm
import json
import pickle


token = ''  # your access token to Llama2

model_id = 'meta-llama/Llama-2-13b-chat-hf'
model_id = "/root/.cache/huggingface/models--meta-llama--Llama-2-13b-chat-hf"
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

    
class NarrationDataset(torch.utils.data.Dataset):
    def __init__(self, clip_narrations):
        self.prompt = """<s>[INST] <<SYS>>
You are an AI Assistant and always write the output of your response in JSON. I will provide you with a series of narrations that depict my behavior. You should generate one QA pair based on the narrations in the format of {\"Q\": <question>, \"A\": <answer>}. In the narrations, \"C\" represents me, and \"O\" represents someone else. Use as much information as possible from narrations to generate the question, and the question you generate should be able to be answered using the information provided in the narrations. The question should be in the past tense. The question should be within 10 words, and the answer should be within 5 words.
<</SYS>>

C pours hot water from the frying pan in his left hand into the bowl in his right hand. [/INST] {\"Q\": \"What did I pour in the bowl?\", \"A\": \"boiling water\"} </s>
<s>[INST] C searches through the cabinet. C closes the cabinet. C picks the tin from the cabinet. C places the tin on the counter. [/INST] {\"Q\": \"Where was the tin before I took it?\", \"A\": \"at the cabinet\"} </s>
<s>[INST] C turns on sink knob. C washes the cucumber on the sink. C turns off sink knob. [/INST] {\"Q\": \"Did I wash the cucumber?\", \"A\": \"yes\"} </s>
"""
        self.narrations = self._prepare(clip_narrations)

    def _prepare(self, clip_narrations):
        sampled_narrations = []
        for c in clip_narrations:
            clip_uid = c['clip_uid']
            narration_pass = c['narration_pass']
            narrations = c['narrations']

            idx = 0
            while idx < len(narrations):
                sampled = self._sample_narrations(narrations, start=idx)
                start_sec = sampled[0]['timestamps'][0]
                end_sec = sampled[-1]['timestamps'][1]
                narration_texts = ' '.join([n['narration_text'] for n in sampled])
                lm_input = self.prompt + "<s>[INST] " + narration_texts + "[/INST]\n"
                sampled_narrations.append({
                    'clip_uid': clip_uid,
                    'narration_pass': narration_pass,
                    'narrations': narration_texts,
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'lm_input': lm_input,
                    'n_narration': len(sampled)
                })
                idx += len(sampled)
            
        return sampled_narrations

    def _sample_narrations(self, narrations, start, max_n=5, max_timespan=30):
        end = min(len(narrations), start+max_n)
        while start < end - 1 and max_timespan <= narrations[end-1]['timestamps'][1] - narrations[start]['timestamps'][0]:
            end -= 1
        end = np.random.choice(np.arange(start, end), 1)[0] + 1
        return narrations[start:end]
    
    def __len__(self):
        return len(self.narrations)
    
    def __getitem__(self, idx):
        return self.narrations[idx]['lm_input']
    
    def get_clip_uid(self, idx):
        return self.narrations[idx]['clip_uid']
    
    def get_start_sec(self, idx):
        return self.narrations[idx]['start_sec']

    def get_end_sec(self, idx):
        return self.narrations[idx]['end_sec']
    
    def get_narrations(self, idx):
        return self.narrations[idx]['narrations']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=int, default=None)
    parser.add_argument('-end', type=int, default=None)
    args = parser.parse_args()

    with open('em_train_narrations.pkl', 'rb') as f:
        clip_narrations = pickle.load(f)
    print(len(clip_narrations))
    os.makedirs('tmp', exist_ok=True)

    if args.start is not None and args.end is not None:
        clip_narrations = clip_narrations[args.start:args.end]
        save_path = f'tmp/annotations.EgoTimeQA_{args.start}_{args.end}.json'
    else:
        save_path = 'annotations.EgoTimeQA.json'

    dataset = NarrationDataset(clip_narrations)

    errors = 0
    pbar = tqdm(total=len(dataset))
    res = []
    for idx, out in enumerate(pipeline(
        dataset,
        batch_size=32,
        do_sample=True,
        temperature=0.5,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        return_full_text=False,
    )):
        pbar.set_description(f'Errors: {errors}')
        pbar.update(1)
        gen_result = out[0]['generated_text']
        try:  # may not generate in JSON format
            qa = literal_eval(gen_result)
            question = qa['Q']
            answer = qa['A']
            start_sec = dataset.get_start_sec(idx)
            end_sec = dataset.get_end_sec(idx)
            start_frame = int(start_sec * 30)
            end_frame = int(end_sec * 30)
            res.append({
                'video_id': dataset.get_clip_uid(idx),
                'sample_id': None,
                'answer': answer,
                'question': question,
                'start_sec': start_sec,
                'end_sec': end_sec,
                "moment_start_frame": start_frame,
                "moment_end_frame": end_frame,
                'narrations': dataset.get_narrations(idx)
            })
        except:
            errors += 1
            continue 
    
    with open(save_path, 'w') as f:
        json.dump(res, f)
