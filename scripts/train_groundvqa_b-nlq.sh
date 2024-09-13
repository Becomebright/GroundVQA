# train with NLQv2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py \
    model=groundvqa_b \
    'dataset.nlq_train_splits=[NLQ_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=16 \
    trainer.find_unused_parameters=True \
    trainer.gpus=8
