# train with QaEgo4D
CUDA_VISIBLE_DEVICES=6,7 python run.py \
    model=groundvqa_s \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    dataset.batch_size=64 \
    trainer.gpus=2