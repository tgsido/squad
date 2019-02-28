rm "generated_train_bert_embeddings"
rm "generated_dev_bert_embeddings"
rm "generated_test_bert_embeddings"

python generate_bert_embeddings.py --data_type "train"
python generate_bert_embeddings.py --data_type "dev"
python generate_bert_embeddings.py --data_type "test"
