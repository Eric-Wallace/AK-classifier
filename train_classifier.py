from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# build dataset
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
tokenizer.pad_token_id = 1
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = load_dataset('json', data_files={'train': 'data/tweets_dataset_train.jsonl', 'test': 'data/tweets_dataset_test.jsonl'})
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512) #1300
tokenized_datasets = dataset.map(preprocess_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("facebook/galactica-125m", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    fp16=True,
    save_strategy = "no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model('./results')

# get eval predictions
predictions = trainer.predict(tokenized_datasets["test"])
positive_preds = predictions.predictions[:,1] #np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# plot ROC curve of model performance
fpr, tpr, _ = roc_curve(labels, positive_preds)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="#FF7974", lw=3, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig('data/roc_plot.png')