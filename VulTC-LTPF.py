import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, matthews_corrcoef
import sys
import collections
import logging
import re
import random

logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 32
num_class = 31
max_seq_l = 512
lr = 5e-5
num_epochs = 5000
use_cuda = True
model_name = "codet5"
pretrainedmodel_path = "E:\\models\\codet5-base"
early_stop_threshold = 10

from openprompt.data_utils import InputExample

classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20',
    'CWE-416', 'CWE-190', 'CWE-200', 'CWE-399', 'CWE-264',
    'CWE-120', 'CWE-362', 'CWE-400', 'CWE-401', 'CWE-189',
    'CWE-617', 'CWE-835', 'CWE-772', 'CWE-287', 'CWE-369',
    'CWE-22', 'CWE-415', 'CWE-674', 'CWE-122', 'CWE-254',
    'CWE-834', 'CWE-770', 'CWE-295', 'CWE-74', 'CWE-310',
    'Remain'
]

tail_classes = [
    'CWE-362', 'CWE-400', 'CWE-401', 'CWE-189', 'Remain'
    'CWE-617', 'CWE-835', 'CWE-772', 'CWE-287', 'CWE-369',
    'CWE-22', 'CWE-415', 'CWE-674', 'CWE-122', 'CWE-254',
    'CWE-834', 'CWE-770', 'CWE-295', 'CWE-74', 'CWE-310'
]

tail_cwe_ids = list(range(11, 31))

def read_prompt_examples(filename):

    examples = []
    data = pd.read_excel(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    cwe = data['cwe_ids_tip'].tolist()
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(cwe[idx]),
            )
        )
    return examples

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate

template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description:" {"placeholder":"text_b"} {"soft":"Classify the cwe:"} {"mask"}'

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)

from openprompt.prompts import ManualVerbalizer

myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    'CWE-119': ['buffer overflow', 'memory violation'],
                                    'CWE-125': ['out-of-bounds read', 'information leak'],
                                    'CWE-787': ['out-of-bounds write', 'memory corruption'],
                                    'CWE-476': ['null pointer dereference', 'null access'],
                                    'CWE-20': ['input validation', 'sanitization error'],
                                    'CWE-416': ['use after free', 'dangling pointer'],
                                    'CWE-190': ['integer overflow', 'arithmetic overflow'],
                                    'CWE-200': ['information disclosure', 'data leak'],
                                    'CWE-399': ['resource management error', 'resource handling'],
                                    'CWE-264': ['permissions and access control', 'authorization'],
                                    'CWE-120': ['buffer overflow', 'memory corruption'],
                                    'CWE-362': ['race condition', 'concurrency issue'],
                                    'CWE-400': ['resource exhaustion', 'denial of service'],
                                    'CWE-401': ['memory leak', 'resource leak'],
                                    'CWE-189': ['integer underflow', 'arithmetic underflow'],
                                    'CWE-617': ['uncontrolled resource', 'assertion failure'],
                                    'CWE-835': ['infinite loop', 'unbounded loop'],
                                    'CWE-772': ['missing release of resource', 'handle leak'],
                                    'CWE-287': ['authentication failure', 'login bypass'],
                                    'CWE-369': ['divide by zero', 'arithmetic exception'],
                                    'CWE-22': ['path traversal', 'file access vulnerability'],
                                    'CWE-415': ['double free', 'multiple memory release'],
                                    'CWE-674': ['uncontrolled recursion', 'stack overflow'],
                                    'CWE-122': ['heap-based buffer overflow', 'dynamic memory error'],
                                    'CWE-254': ['protection mechanism failure', 'security mechanism bypass'],
                                    'CWE-834': ['excessive iteration', 'inefficiency'],
                                    'CWE-770': ['unbounded resource allocation', 'resource allocation flaw'],
                                    'CWE-295': ['improper certificate validation', 'certificate verification'],
                                    'CWE-74': ['code injection', 'injection vulnerability'],
                                    'CWE-310': ['cryptographic issue', 'insecure cryptography'],
                                    'Remain': ['general security issue', 'other threat']
                                })

from openprompt import PromptForClassification

prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)

def replace_variable_names(code):
    variable_names = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    keywords = set(['int', 'float', 'if', 'else', 'for', 'while', 'return', 'void', 'char', 'double', 'include', 'define'])
    variable_names = [var for var in set(variable_names) if var not in keywords]
    new_code = code
    for var in variable_names:
        new_var = var + '_' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
        new_code = re.sub(r'\b{}\b'.format(var), new_var, new_code)
    return new_code

def insert_redundant_code(code):
    lines = code.split('\n')
    insert_position = random.randint(0, len(lines))
    redundant_line = '// ' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=20))
    lines.insert(insert_position, redundant_line)
    new_code = '\n'.join(lines)
    return new_code

from tqdm.auto import tqdm

def test(prompt_model, test_dataloader, name, return_tail_class_errors=False):
    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    allpreds = []
    alllabels = []

    labelwise_correct = collections.defaultdict(int)
    labelwise_total = collections.defaultdict(int)

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']

            progress_bar.update(1)
            alllabels.extend(labels.cpu().tolist())
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            allpreds.extend(preds)

            for label, pred in zip(labels.cpu().tolist(), preds):
                if label == pred:
                    labelwise_correct[label] += 1
                labelwise_total[label] += 1

        acc = accuracy_score(alllabels, allpreds)

        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)

        labelwise_accuracy = []
        for idx, class_name in enumerate(classes):
            total = labelwise_total[idx]
            correct = labelwise_correct[idx]
            accuracy = correct / total if total > 0 else 0
            labelwise_accuracy.append((class_name, accuracy))

        accuracy_df = pd.DataFrame(labelwise_accuracy, columns=['Class', 'Accuracy'])

        output_file = f'./changresults/{name}_per_class_accuracy.xlsx'
        accuracy_df.to_excel(output_file, index=False)

        logging.info(f"Per-class accuracy saved to {output_file}")

        with open(os.path.join('./results', "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join('./results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:

            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')

        logging.info("Evaluation Results - Acc: {:.4f}, Precision (Macro): {:.4f}, Recall (Macro): {:.4f}, "
                     "Recall (Weighted): {:.4f}, F1 (Weighted): {:.4f}, F1 (Macro): {:.4f}, MCC: {:.4f}".format(
                         acc, precisionma, recallma, recallwei, f1wei, f1ma, mcc))

        if return_tail_class_errors:
            tail_class_errors = {}
            for idx in tail_cwe_ids:
                total = labelwise_total[idx]
                incorrect = total - labelwise_correct[idx]
                error_rate = incorrect / total if total > 0 else 0
                tail_class_errors[idx] = error_rate
            return acc, precisionma, recallma, f1wei, f1ma, tail_class_errors
        else:
            return acc, precisionma, recallma, f1wei, f1ma

from openprompt import PromptDataLoader

train_dataset = read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\.xlsx")

train_dataloader = PromptDataLoader(dataset=train_dataset, template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    batch_size=batch_size, shuffle=True,
                                    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                    decoder_max_length=3)
validation_dataloader = PromptDataLoader(dataset=read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\valid.xlsx"), template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         batch_size=batch_size, shuffle=True,
                                         teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                         decoder_max_length=3)
test_dataloader = PromptDataLoader(dataset=read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\test.xlsx"), template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                   batch_size=batch_size, shuffle=False,
                                   teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                   decoder_max_length=3)

num_training_steps = num_epochs * len(train_dataloader)
scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0.1*num_training_steps,
                                             num_training_steps=num_training_steps)
scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0.1*num_training_steps,
                                             num_training_steps=num_training_steps)

output_dir = "code log"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

progress_bar = tqdm(range(num_training_steps))
bestmetric = 0
bestepoch = 0
early_stop_count = 0

for epoch in range(num_epochs):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):

        if use_cuda:
            inputs = inputs.cuda()

        logits = prompt_model(inputs)

        labels = inputs['tgt_text'].cuda()

        loss = loss_func(logits, labels)
        try:
            loss.backward()
        except Exception as e:
            logging.error(f"Loss computation error at epoch {epoch}, step {step}: {e}")
            exit()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        scheduler1.step()
        optimizer2.step()
        optimizer2.zero_grad()
        scheduler2.step()
        progress_bar.update(1)
    avg_loss = tot_loss / (step + 1)
    logging.info("Epoch {}, average loss: {:.4f}".format(epoch, avg_loss))

    logging.info('Epoch {} - Validation Phase'.format(epoch))
    acc, precision, recall, f1wei, f1mi, tail_class_errors = test(prompt_model, validation_dataloader, name="dev", return_tail_class_errors=True)

    k = 15
    augmented_examples = []
    tail_error_info = []
    for idx, cwe_id in enumerate(tail_cwe_ids):
        error_rate = tail_class_errors.get(cwe_id, 0)
        n_augment = int(k * error_rate)
        tail_error_info.append((classes[cwe_id], error_rate, n_augment))
        if n_augment > 0:
            tail_examples = [ex for ex in train_dataset if int(ex.tgt_text) == cwe_id]
            if not tail_examples:
                continue
            for _ in range(n_augment):
                original_example = random.choice(tail_examples)
                if random.random() < 0.5:
                    new_code = replace_variable_names(original_example.text_a)
                else:
                    new_code = insert_redundant_code(original_example.text_a)
                new_example = InputExample(
                    guid=len(train_dataset) + len(augmented_examples),
                    text_a=' '.join(new_code.split(' ')[:384]),
                    text_b=' '.join(original_example.text_b.split(' ')[:64]),
                    tgt_text=original_example.tgt_text
                )
                augmented_examples.append(new_example)

    for class_name, error_rate, n_augment in tail_error_info:
        logging.info("Tail Class: {}, Error Rate: {:.4f}, Samples to Augment: {}".format(class_name, error_rate, n_augment))

    if augmented_examples:
        train_dataset.extend(augmented_examples)
        train_dataloader = PromptDataLoader(
            dataset=train_dataset,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )
    logging.info("Data augmentation completed for epoch {}, added {} samples.".format(epoch, len(augmented_examples)))

    if acc > bestmetric:
        bestmetric = acc
        bestepoch = epoch
        torch.save(prompt_model.state_dict(), f"{output_dir}/best.ckpt")
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count == early_stop_threshold:
            logging.info("Early stopping at epoch {}".format(epoch))
            break

logging.info('Testing with best model from epoch {}'.format(bestepoch))
prompt_model.load_state_dict(torch.load(os.path.join(output_dir, "best.ckpt"), map_location=torch.device('cuda:0')))

acc, precisionma, recallma, f1wei, f1ma = test(prompt_model, test_dataloader, name="test")
