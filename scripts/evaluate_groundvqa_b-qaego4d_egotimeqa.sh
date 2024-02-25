# QAEgo4D-Close test set
for SEED in 0 1111 2222 3333 4444
do
    python run.py \
        model=groundvqa_b \
        'dataset.qa_train_splits=[QaEgo4D_train]' \
        'dataset.test_splits=[QaEgo4D_test_close]' \
        dataset.batch_size=32 \
        +trainer.test_only=True \
        '+trainer.checkpoint_path="checkpoints/GroundVQA_B-QaEgo4D_EgoTimeQA-COV-test_ROUGE=30.4.ckpt"' \
        +trainer.random_seed=$SEED
done

# QAEgo4D-Open test set
python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[QaEgo4D_test]' \
    dataset.batch_size=32 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-QaEgo4D_EgoTimeQA-COV-test_ROUGE=30.4.ckpt"'

# NLQv2 val set
python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=32 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="checkpoints/GroundVQA_B-QaEgo4D_EgoTimeQA-COV-val_R1_03=25.6.ckpt"' \
    trainer.load_nlq_head=True 