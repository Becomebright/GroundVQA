# NLQv2 val set
CUDA_VISIBLE_DEVICES=1 python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=32 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path=""' \
    trainer.load_nlq_head=True
