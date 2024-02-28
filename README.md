# GroundVQA

Official PyTorch code of "Grounded Question-Answering in Long Egocentric Videos", *CVPR* 2024.

[[Project page]](https://dszdsz.cn/GroundVQA/index.html) [[Paper]](https://arxiv.org/abs/2312.06505)

## News

[Feb 2024] We update the CloseQA test set with rigorous human verification. The benchmark results will be updated in our paper shortly.

## Abstract

Existing approaches to video understanding, mainly designed for short videos from a third-person perspective, are limited in their applicability in certain fields, such as robotics. In this paper, we delve into **open-ended question-answering (QA) in long, egocentric videos**, which allows individuals or robots to inquire about their own past visual experiences. 

This task presents **unique challenges**, including the complexity of temporally grounding queries within extensive video content, the high resource demands for precise data annotation, and the inherent difficulty of evaluating open-ended answers due to their ambiguous nature. 

Our proposed approach tackles these challenges by 

- **GroundVQA**: integrating query grounding and answering within a unified model to reduce error propagation;
- **EgoTimeQA**: employing large language models for efficient and scalable data synthesis;
- **QaEgo4D**$`_\texttt{close}`$: introducing a close-ended QA task for evaluation, to manage answer ambiguity.

Extensive experiments demonstrate the effectiveness of our method, which also achieves state-of-the-art performance on the QaEgo4D and Ego4D-NLQ benchmarks.

## Directory Structure

```
.
|-- checkpoints       provided model checkpoints
|-- config            configs of models and datasets
|-- data              processed dataset and video features
|-- eval.py           code for evaluating QaEgo4D performance
|-- eval_nlq.py				code for evaluating NLQ performance
|-- model             code for model, dataset, and training
|-- requirements.txt  list of packages for building the Python environment
|-- run.py            entry code
|-- scripts           scripts for training and evaluation
`-- utils             code for generating OpenQA and CloseQA data from Ego4D narrations
```

## Preparation

Our setup: Ubuntu 20.04, CUDA 12.2, 8x Nvidia A100 (80GB)

- Clone this repo: `https://github.com/Becomebright/GroundVQA.git`
- Create the conda environment: `conda create -n groundvqa python=3.9 -y && conda activate groundvqa`
- Install packages: `pip install -r requirements.txt`
- Compile `nms_1d_cpu` following [here](https://github.com/happyharrycn/actionformer_release/blob/main/INSTALL.md)
- Download the data, video feature, and model checkpoints from [Huggingface](https://huggingface.co/Becomebright/GroundVQA)
  - **data:** unzip `data.zip` under the project's root directory.
  - **video feature:** merge the files `cat egovlp_internvideoa* > egovlp_internvideo.hdf5` and put it under `data/unified/`
  - **model checkpoints**: put them under `checkpoints/`

| Model                           | Data                                                         | Task                   | NLQ$`_\texttt{v2}`$                                          | QaEgo4D                                                      | Cost$`^{*}`$ |
| ------------------------------- | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| $`\text{GroundVQA}_\texttt{S}`$ | QaEgo4D                                                      | CloseQA+OpenQA+VLG | [[val_R1_03=11.0]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_S-QaEgo4D-COV-val_R1_03%3D11.0.ckpt) | [[test_ROUGE=29.0]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_S-QaEgo4D-COV-test_ROUGE%3D29.0.ckpt) | 7              |
| $`\text{GroundVQA}_\texttt{S}`$ | QaEgo4D+EgoTimeQA                                          | CloseQA+OpenQA+VLG | [[val_R1_03=23.3]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_S-QaEgo4D_EgoTimeQA-COV-val_R1_03%3D23.3.ckpt) | [[test_ROUGE=30.2]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_S-QaEgo4D_EgoTimeQA-COV-test_ROUGE%3D30.2.ckpt) | 150            |
| $`\text{GroundVQA}_\texttt{B}`$ | QaEgo4D+EgoTimeQA                                          | CloseQA+OpenQA+VLG | [[val_R1_03=25.6]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_B-QaEgo4D_EgoTimeQA-COV-val_R1_03%3D25.6.ckpt) | [[test_ROUGE=30.4]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_B-QaEgo4D_EgoTimeQA-COV-test_ROUGE%3D30.4.ckpt) | 350            |
| $`\text{GroundVQA}_\texttt{B}`$ | NLQ$`_\texttt{v2}`$+NaQ $\rightarrow$ NLQ$`_\texttt{v2}`$$`^{**}`$ | VLG                    | [[val_R1_03=29.7]](https://huggingface.co/Becomebright/GroundVQA/blob/main/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03%3D29.7.ckpt) | -                                                            | 700            |

\* The training costs counted by GPU hours.

** Pre-trained on NLQ$`_\texttt{v2}`$ and NaQ, and further fine-tuned on NLQ$`_\texttt{v2}`$​.

## Training

```bash
# train GroundVQA_S on QaEgo4D
bash scripts/train_groundvqa_small-qaego4d.sh

# train GroundVQA_S on QaEgo4D and EgoTimeQA
bash scripts/train_groundvqa_small-qaego4d_egotimeqa.sh
    
# train GroundVQA_B on QaEgo4D and EgoTimeQA
bash scripts/train_groundvqa_base-qaego4d_egotimeqa.sh
```

## Evaluation

```bash
# evaluate GroundVQA_S train on QaEgo4D
bash scripts/evaluate_groundvqa_s-qaego4d.sh

# evaluate GroundVQA_S train on QaEgo4D and EgoTimeQA
bash scripts/evaluate_groundvqa_s-qaego4d_egotimeqa.sh

# evaluate GroundVQA_B train on QaEgo4D and EgoTimeQA
bash scripts/evaluate_groundvqa_b-qaego4d_egotimeqa.sh

# evaluate GroundVQA_B train on NLQv2 and NaQ and further fine-tuned on NLQv2
bash scripts/evaluate_groundvqa_b-nlq_naq.sh
```

## Generate OpenQA data

Download the processed Ego4D narrations [[em_train_narrations.pkl]](https://huggingface.co/Becomebright/GroundVQA/blob/main/em_train_narrations.pkl)

Put it under `utils/generate_open_qa/`

Generate QAs in parallel on multiple GPUs (*e.g.*, 2)

```bash
cd utils/generate_open_qa

# GPU-0
CUDA_VISIBLE_DEVICES=0 python generate.py -start 0 -end 5000

# GPU-1
CUDA_VISIBLE_DEVICES=1 python generate.py -start 5000 -end 11000  # 10777 clips in total
```

Merge the results and normalize the duration of temporal windows

```bash
python merge.py
```

## Generate CloseQA data

```bash
cd utils/generate_close_qa
python generate.py
```

The above script produce wrong answers for EgoTimeQA using a single GPU.

You can also conduct generation on multiple GPUs or generate wrong answers for QaEgo4D.

## Citation

```latex
@inproceedings{di2023groundvqa,
  title={Grounded Question-Answering in Long Egocentric Videos},
  author={Di, Shangzhe and Xie, Weidi},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgements

Our code is based on [QaEgo4D](https://github.com/lbaermann/qaego4d), [GroundNLQ](https://github.com/houzhijian/GroundNLQ), and [ActionFormer](https://github.com/happyharrycn/actionformer_release).