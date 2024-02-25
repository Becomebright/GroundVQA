# NLQv2 val set
python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=32 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-NLQ_NaQ-finetune_NLQ-VLG-val_R1_03=29.7.ckpt"' \
    trainer.load_nlq_head=True 