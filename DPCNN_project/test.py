from utils import *
from model import *
from dataloader import *
from fastNLP import Tester
from fastNLP.core.metrics import AccuracyMetric
# from fastNLP.core.utils import load_pickle
from fastNLP.io.model_io import ModelLoader

# define model
model=DPCNN(max_features=len(vocab),word_embedding_dimension=word_embedding_dimension,max_sentence_length = max_sentence_length,num_classes=num_classes,weight=weight)

# load checkpoint to model
# load_model = load_pickle(pickle_path=pickle_path, file_name='model_ckpt_100.pkl')
load_model_ = ModelLoader()
load_model = load_model_.load_pytorch_model('./model_backup/train_best_model.pkl')
# print(load_model)
# print(type(load_model))
# use Tester to evaluate
tester=Tester(data=dataset_test,model=load_model,metrics=AccuracyMetric(pred="predict",target="label_seq"),batch_size=64)
acc=tester.test()
print(acc)
