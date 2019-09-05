from utils import *
from model import *
from dataloader import *
from fastNLP import Trainer
from copy import deepcopy
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core.optimizer import Adam
# from fastNLP.core.utils import save_pickle
from fastNLP.io.model_io import ModelSaver

# load model
model=DPCNN(max_features=len(vocab)+1,word_embedding_dimension=word_embedding_dimension,max_sentence_length = max_sentence_length,num_classes=num_classes,weight=weight)

# define loss and metric
loss = CrossEntropyLoss(pred="output",target="label_seq")
metric = AccuracyMetric(pred="predict", target="label_seq")

# train model with train_data,and val model with test_data
# embedding=300 gaussian init，weight_decay=0.0001, lr=0.001，epoch=5
trainer=Trainer(model=model,train_data=dataset_train,dev_data=dataset_test,loss=loss,metrics=metric,save_path='new_model.pkl',batch_size=64,n_epochs=5,optimizer=Adam(lr=0.001, weight_decay=0.0001))
trainer.train()

# save pickle
# saver = ModelSaver("./result/model_ckpt_100.pkl")
# saver.save_pytorch(model)
# save_pickle(model,pickle_path=pickle_path,file_name='new_model.pkl')
