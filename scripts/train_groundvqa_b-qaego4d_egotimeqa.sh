# train with QaEgo4D + EgoTimeQA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train,EgoTimeQA]' \
    dataset.batch_size=16 \
    trainer.gpus=8
