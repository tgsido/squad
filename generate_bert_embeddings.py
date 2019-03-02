import torch
import time

def get_embeddings(data_type, ids, max_context_len, max_question_len):
    start = time.time()
    ## SET DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_CONTEXT_LEN = max_context_len
    MAX_QUESTION_LEN = max_question_len
    MAX_SEQ_LENGTH = MAX_CONTEXT_LEN + MAX_QUESTION_LEN

    ids = ids.tolist()

    directory = "/datasquad/" + data_type + "_bert_embeddings/"
   # print("directory: ", directory)

    batch_size = len(ids)
    embeddings = torch.zeros((batch_size, MAX_SEQ_LENGTH, 768), device=device)
    for i, id in enumerate(ids):
        file_path = directory + str(id) + ".pt"
       # print("file_path: ", file_path)
        id_embedding = torch.load(file_path)
        embeddings[i,:,:] = id_embedding
    end = time.time()
    #print("time to fetch bert embeddings: ", end - start)

    return embeddings
