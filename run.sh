python ./openspeech_cli/hydra_train.py \
    dataset=ksponspeech \
    dataset.dataset_path=/home/PLASS-war/project/openspeech1/dataset/kSponSpeech/KsponSpeech \
    dataset.manifest_file_path=/home/PLASS-war/project/openspeech1/dataset/kSponSpeech/KsponSpeech_train_mani/transcripts.txt \
    dataset.test_manifest_dir=/home/PLASS-war/project/openspeech1/dataset/kSponSpeech/KsponSpeech_scripts \
    dataset.test_dataset_path=/home/PLASS-war/project/openspeech1/dataset/kSponSpeech/KsponSpeech_eval \
    trainer.seed=42 \
    trainer.batch_size=72 \
    trainer.num_workers=24 \
    model.encoder_dim=256 \
    model.num_encoder_layers=16 \
    model.num_attention_heads=4 \
    tokenizer=kspon_character \
    model=conformer_lstm \
    audio=fbank \
    lr_scheduler=transformer \
    trainer=gpu \
    criterion=cross_entropy \
