# QAEgo4D-Close test set
for SEED in 0 1111 2222 3333 4444
do
    python run.py \
        model=groundvqa_s \
        'dataset.qa_train_splits=[QaEgo4D_train]' \
        'dataset.test_splits=[QaEgo4D_test_close]' \
        dataset.batch_size=64 \
        +trainer.test_only=True \
        '+trainer.checkpoint_path="checkpoints/GroundVQA_S-QaEgo4D-COV-test_ROUGE=29.0.ckpt"' \
        +trainer.random_seed=$SEED
done

# QAEgo4D-Open test set
python run.py \
    model=groundvqa_s \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[QaEgo4D_test]' \
    dataset.batch_size=64 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_S-QaEgo4D-COV-test_ROUGE=29.0.ckpt"'

# NLQv2 val set
python run.py \
    model=groundvqa_s \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=64 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_S-QaEgo4D-COV-val_R1_03=11.0.ckpt"' \
    trainer.load_nlq_head=True
