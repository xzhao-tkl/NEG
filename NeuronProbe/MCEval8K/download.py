import os
import csv
import random
import datasets
import kagglehub
import pandas as pd

from conllu import parse
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from urllib.request import urlopen
from datasets import get_dataset_config_names
from datasets import Dataset, Features, Value, Sequence
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

BASE_DIR = Path("./")
DATASET_CACHE_ROOT = Path("<path/to/your/dataset/cache/root>")  # Replace with your actual path

universal_pos_tags = [
    "ADJ", "ADP", "ADV", "AUX", 
    "CCONJ", "DET", "INTJ", "NOUN", 
    "NUM", "PART", "PRON", "PROPN", 
    "PUNCT", "SCONJ", "SYM", "VERB", 
]


def option_sampling(dataset, num_sample, labels, label_column='label'):
    num_sample_per_label = []
    for i in range(len(labels)):
        num_sample_left = num_sample - sum(num_sample_per_label)
        num_sample_per_label.append(int(num_sample_left/(len(labels) - i)))
    all_ds = []
    for i, label in enumerate(labels):
        _ds = dataset.filter(lambda x: x[label_column] == label)
        _ds = _ds.shuffle(seed=42).select(range(num_sample_per_label[i]))
        all_ds.append(_ds)
    ds = datasets.concatenate_datasets(all_ds)
    return ds

def train_test_sampling(dataset, num_train, labels, label_column='label'):
    num_sample_per_label = []
    for i in range(len(labels)):
        num_sample_left = num_train - sum(num_sample_per_label)
        num_sample_per_label.append(int(num_sample_left/(len(labels) - i)))
    
    all_train_ds, all_test_ds = [], []
    for i, label in enumerate(labels):
        _ds = dataset.filter(lambda x: x[label_column] == label)
        selected_indices = set(range(num_sample_per_label[i]))
        unselected_indices = set(range(len(_ds) - num_sample_per_label[i]))
        train_ds = _ds.shuffle(seed=42).select(selected_indices)
        test_ds = _ds.shuffle(seed=42).select(unselected_indices)
        all_train_ds.append(train_ds)
        all_test_ds.append(test_ds)
    train_ds = datasets.concatenate_datasets(all_train_ds)
    test_ds = datasets.concatenate_datasets(all_test_ds)
    return train_ds, test_ds

def dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns):
    train_ds.to_pandas()[ordered_columns].to_json(data_dir / "train.json", index=False, orient='records', indent=True)
    valid_ds.to_pandas()[ordered_columns].to_json(data_dir / "valid.json", index=False, orient='records', indent=True)
    test_ds.to_pandas()[ordered_columns].to_json(data_dir / "test.json", index=False, orient='records', indent=True)
    
def process_agnews(data_dir):
    dataset = load_dataset("fancyzhx/ag_news")
    train_ds = option_sampling(dataset['train'], 7000, range(4))
    test_ds = option_sampling(dataset['test'], 1000, range(4))
    train_ds, valid_ds = train_test_sampling(train_ds, 6000, range(4))

    assert len(set(Counter(train_ds['label']).values())) == 1
    assert len(set(Counter(valid_ds['label']).values())) == 1
    assert len(set(Counter(test_ds['label']).values())) == 1

    # train_ds.to_csv(data_dir / "train.csv", index=False)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False)
    # test_ds.to_csv(data_dir / "test.csv", index=False)
    ordered_columns = ['text', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)
    

def process_imdb(data_dir):
    dataset = load_dataset("stanfordnlp/imdb")
    train_ds = option_sampling(dataset['train'], 7000, range(2))
    test_ds = option_sampling(dataset['test'], 1000, range(2))
    train_ds, valid_ds = train_test_sampling(train_ds, 6000, range(2))

    assert len(set(Counter(train_ds['label']).values())) == 1
    assert len(set(Counter(valid_ds['label']).values())) == 1
    assert len(set(Counter(test_ds['label']).values())) == 1

    # train_ds.to_csv(data_dir / "train.csv", index=False)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False)
    # test_ds.to_csv(data_dir / "test.csv", index=False)
    ordered_columns = ['text', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_amazonreview(data_dir):
    config2ds = {}
    AMAZON_CACHE_DIR = DATASET_CACHE_ROOT / "McAuley-Lab___amazon-reviews-2023" / "cache"
    configs = [config for config in get_dataset_config_names("McAuley-Lab/Amazon-Reviews-2023") if config.startswith("raw_review")]
    for config in configs:
        cache_dir = AMAZON_CACHE_DIR / config 
        if cache_dir.exists() and any(cache_dir.iterdir()):
            ds = datasets.load_from_disk(cache_dir.as_posix())
        else:
            ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", config, trust_remote_code=True)
            ds = ds['full'].shuffle(seed=42).select(range(10000))
            ds.save_to_disk(cache_dir.as_posix())
        config2ds[config] = ds

    all_ds, num_items = [], []
    for i, config in enumerate(config2ds):
        num_item = int((8000-sum(num_items))/(len(config2ds)-i)) if i < len(config2ds)-1 else 8000-sum(num_items)
        num_items.append(num_item)
        ds = config2ds[config].filter(lambda x: len(x['text'])>100)
        assert len(Counter(ds['rating'])) == 5
        all_ds.append(ds)

    ds = datasets.concatenate_datasets(all_ds)
    ds = option_sampling(ds, 8000, range(1, 6), label_column='rating')
    ds = ds.rename_column('rating', 'label')
    removable_columns = [column for column in ds.column_names if column not in ['label', 'text']]
    ds = ds.remove_columns(removable_columns)
    train_ds, test_ds = train_test_sampling(ds, 6000, range(1, 6))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, range(1, 6))

    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)
    
    ordered_columns = ['text', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def weighted_sample(items, weights, num_sample):
    samples = []
    while len(samples) < num_sample:
        sample = random.choices(items, weights=weights, k=1)[0]
        if sample not in samples:
            samples.append(sample) 
    return samples

def process_myriadlama(data_dir):
    rel2objs, obj2label = {}, {}
    ds = datasets.load_dataset('iszhaoxin/MyriadLAMA', trust_remote_code=True).shuffle(seed=42)
    for rel_uri, obj_uris, obj_ents in zip(ds['train']['rel_uri'], ds['train']['obj_uris'], ds['train']['obj_ents']):
        if rel_uri not in rel2objs:
            rel2objs[rel_uri] = {}
        for obj_uri in obj_uris:
            if obj_uri not in rel2objs[rel_uri]:
                rel2objs[rel_uri][obj_uri] = 1
            else:
                rel2objs[rel_uri][obj_uri] += 1
        for obj_uri, obj_ent in zip(obj_uris, obj_ents):
            obj2label[obj_uri] = obj_ent

    ds = ds.filter(lambda x: x['is_manual'] == True)
    def create_label(item):
        cands = rel2objs[item['rel_uri']].keys()
        non_ans = list(set(cands) - set(item['obj_uris']))
        non_ans_weights = [rel2objs[item['rel_uri']][obj] for obj in non_ans]
        options = weighted_sample(items=non_ans, weights=non_ans_weights, num_sample=3)
        assert len(set(options)) == 3
        answer = random.sample(item['obj_uris'], 1)[0]
        options.append(answer)
        assert len(set(options)) == 4
        random.shuffle(options)
        item['options'] = [obj2label[option] for option in options]
        idx = options.index(answer)
        item['label'] = idx
        item['text'] = item['template'].replace('[X]', item['sub_ent'])
        return item
    ds = ds.map(create_label)

    removable_columns = [column for column in ds['train'].column_names if column not in ['text', 'label', 'options']]
    ds = ds.remove_columns(removable_columns)

    _ds = option_sampling(ds['train'], num_sample=8000, labels=range(4))
    train_ds, test_ds = train_test_sampling(_ds, 6000, labels=range(4))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, labels=range(4))

    ordered_columns = ['text', 'options', 'label']
    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_fever(data_dir):
    ds = datasets.load_dataset('fever/fever', 'v1.0', trust_remote_code=True).shuffle(seed=42)
    def processs_ds(ds):
        df = ds.to_pandas()
        df['label'] = df['label'].map(lambda x: 0 if x=='REFUTES' else 1)
        df = df.drop(columns=[col for col in df.columns if col not in ['claim', 'label']])
        return Dataset.from_pandas(df)

    train_ds = option_sampling(processs_ds(ds['train']), num_sample=6000, labels=range(2))
    dev_ds = option_sampling(processs_ds(ds['labelled_dev']), num_sample=2000, labels=range(2))
    valid_ds, test_ds = train_test_sampling(dev_ds, 1000, labels=range(2))

    ordered_columns = ['claim', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_commensenseqa(data_dir):
    ds = datasets.load_dataset('tau/commonsense_qa', trust_remote_code=True).shuffle(seed=42)
    def create_label(item):
        if item['answerKey'] == '':
            return item
        item['text'] = item['question']
        item['label'] = item['choices']['label'].index(item['answerKey'])
        item['options'] = item['choices']['text']
        return item
    ds = ds.map(create_label)
    ds['train'] = ds['train'].remove_columns([column for column in ds['train'].column_names if column not in ['text', 'options', 'label']])
    ds['validation'] = ds['validation'].remove_columns([column for column in ds['validation'].column_names if column not in ['text', 'options', 'label']])
    train_ds = option_sampling(ds['train'], num_sample=7000, labels=range(5))
    train_ds, valid_ds = train_test_sampling(train_ds, 6000, labels=range(5))
    test_ds = option_sampling(ds['validation'], num_sample=1000, labels=range(5))

    ordered_columns = ['text', 'options', 'label']
    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns, doublequote=True)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns, doublequote=True)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns, doublequote=True)
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)
    return train_ds, valid_ds, test_ds

def process_templama(data_dir):
    ds = datasets.load_dataset('Yova/templama').shuffle(seed=42)
    concat_ds = datasets.concatenate_datasets([ds['train'], ds['validation'], ds['test']])
    df = concat_ds.to_pandas()
    all_years = set(df['date'].tolist())

    texts, optionss, labels = [], [], []
    for query, sdf in df.groupby('query'):
        answer2date = {}
        for answers, date in zip(sdf['answer'], sdf['date']):
            for answer in list(answers):
                answer2date.setdefault(list(answer['original_name'])[0], []).append(date)
        for answer in answer2date:
            options = []
            answer2date[answer] = sorted(answer2date[answer])
            neg_years = all_years.difference(set(answer2date[answer]))
            for i in range(max(3-len(neg_years), 0)):
                neg_years.add(int(2009 - i))
            options.extend(list(neg_years))
            
            random.seed(f"{query}-{answer}")
            options = random.sample(options, k=3)
            
            random.seed(f"{query}-{answer}")
            selected_ans = random.choice(answer2date[answer])
            options.append(selected_ans)
            
            random.seed(f"{query}-{answer}")
            random.shuffle(options)
            label = options.index(selected_ans)
            text = query.replace("_X_", answer)
            texts.append(text)
            optionss.append([str(option) for option in options])
            labels.append(label)
    new_df = pd.DataFrame({'text': texts, 'options': optionss, 'label': labels})

    features = Features({
        'text': Value('string'),   # or 'int64'
        'options': Sequence(Value('string')),  # or 'str'
        'label': Value('int32')     # for boolean columns
    })
    new_ds = Dataset.from_pandas(new_df, features=features)
    new_ds = option_sampling(new_ds, 8000, range(4))
    train_ds, test_ds = train_test_sampling(new_ds, 6000, range(4))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, range(4))

    ordered_columns = ['text', 'options', 'label']
    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_paws(data_dir):
    ds = datasets.load_dataset("google-research-datasets/paws", "labeled_final").shuffle(seed=42)
    ds = ds['train'].remove_columns('id')
    balanced_ds = option_sampling(ds, num_sample=8000, labels=range(2))
    train_ds, test_ds = train_test_sampling(balanced_ds, 6000, range(2))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, range(2))
    
    ordered_columns = ['sentence1', 'sentence2', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_mnli(data_dir):
    ds = datasets.load_dataset("nyu-mll/glue", "mnli").shuffle(seed=42)
    train_ds = option_sampling(ds['train'], num_sample=6000, labels=range(3)).remove_columns(['idx'])
    valid_ds = option_sampling(ds['validation_matched'], num_sample=2000, labels=range(3)).remove_columns(['idx'])
    valid_ds, test_ds = train_test_sampling(valid_ds, num_train=1000, labels=range(3))

    ordered_columns = ['premise', 'hypothesis', 'label']    
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)

def process_swag(data_dir):
    ds = datasets.load_dataset("allenai/swag", "regular").shuffle(seed=42)
    def process_line(item):
        item['text'] = item['startphrase']
        item['options'] = [item['ending0'], item['ending1'], item['ending2'], item['ending3']]
        return item
    processed_ds = ds.map(process_line)
    column_names = processed_ds['train'].column_names
    removeable_columns = [column for column in column_names if column not in ['text', 'options', 'label']]
    processed_ds['train'] = processed_ds['train'].remove_columns(removeable_columns)
    processed_ds['validation'] = processed_ds['validation'].remove_columns(removeable_columns)
    train_ds = option_sampling(processed_ds['train'], num_sample=6000, labels=range(4))
    valid_ds = option_sampling(processed_ds['validation'], num_sample=2000, labels=range(4))
    valid_ds, test_ds = train_test_sampling(valid_ds, num_train=1000, labels=range(4))

    ordered_columns = ['text', 'options', 'label']
    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)
    # return train_ds, valid_ds, test_ds

def process_halueval(data_dir):
    ordered_columns = ['knowledge', 'dialogue', 'response', 'hallucination']
    ds = datasets.load_dataset("pminervini/HaluEval", "dialogue_samples").shuffle(seed=42)
    def process_line(item):
        item['hallucination'] = 1 if item['hallucination'] == 'yes' else 0
        return item
    processed_ds = ds.map(process_line)
    processed_ds = processed_ds.rename_column('dialogue_history', 'dialogue')
    removeable_columns = [
        column for column in processed_ds['data'].column_names 
        if column not in ordered_columns]
    processed_ds['data'] = processed_ds['data'].remove_columns(removeable_columns)
    balanced_ds = option_sampling(processed_ds['data'], num_sample=8000, labels=range(2), label_column='hallucination')
    train_ds, valid_ds = train_test_sampling(balanced_ds, num_train=6000, labels=range(2), label_column='hallucination')
    valid_ds, test_ds = train_test_sampling(valid_ds, num_train=1000, labels=range(2), label_column='hallucination')
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_toxicity(data_dir):
    ds = datasets.load_dataset('jigsaw_toxicity_pred', trust_remote_code=True, data_dir='../raw_data')
    filtered_ds = ds.filter(lambda x: len(x['comment_text'])>100 and len(x['comment_text'])<500)
    filtered_ds['train'] = filtered_ds['train'].rename_column('comment_text', 'text')
    filtered_ds['test'] = filtered_ds['test'].rename_column('comment_text', 'text')
    removeable_columns = [
        column for column in filtered_ds['train'].column_names 
        if column not in ['text', 'toxic']]
    filtered_ds['train'] = filtered_ds['train'].remove_columns(removeable_columns)
    filtered_ds['test'] = filtered_ds['test'].remove_columns(removeable_columns)
    train_ds = option_sampling(filtered_ds['train'], num_sample=7000, labels=range(2), label_column='toxic')
    train_ds, valid_ds = train_test_sampling(train_ds, num_train=6000, labels=range(2), label_column='toxic')
    test_ds = option_sampling(filtered_ds['test'], num_sample=1000, labels=range(2), label_column='toxic')
    ordered_columns = ['text', 'toxic']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_stereoset(data_dir):
    ds1 = datasets.load_dataset('McGill-NLP/stereoset', 'intersentence')
    ds2 = datasets.load_dataset('McGill-NLP/stereoset', 'intrasentence')
    ds = datasets.concatenate_datasets([ds1['validation'], ds2['validation']])

    ordered_columns = ['context', 'sentences', 'stereotype', 'anti-stereotype', 'non-stereotype']
    def process(item):
        gold_labels = item['sentences']['gold_label']
        item['sentences'] = item['sentences']['sentence']
        item['stereotype'] = gold_labels.index(0)
        item['anti-stereotype'] = gold_labels.index(1)
        item['non-stereotype'] = gold_labels.index(2)
        return item
    ds = ds.map(process)
    ds = ds.remove_columns([col for col in ds.column_names if col not in ordered_columns])
    _ds = ds.train_test_split(test_size=1/4)
    _ds2  = _ds['test'].train_test_split(test_size=1/2)
    dump_datasets(data_dir, _ds['train'], _ds2['train'], _ds2['test'], ordered_columns)


def process_multiamazon(data_dir):
    splits = {'train': [], 'test': [], 'validation': []}
    path = kagglehub.dataset_download("mexwell/amazon-reviews-multi")
    ds = datasets.load_dataset('csv', data_files={
        'train': os.path.join(path, 'train.csv'),
        'test': os.path.join(path, 'test.csv'),
        'validation': os.path.join(path, 'validation.csv'),
    })
    ds = ds.filter(lambda x: x['language']!='ja' and len(x['review_body'])>100)
    for lang in ['de', 'fr', 'es', 'en']:
        lang_ds = ds.filter(lambda x: x['language']==lang)
        for split in splits:
            num_sample = 6000/5 if split == 'train' else 1000/5
            _ds = option_sampling(lang_ds[split], num_sample=num_sample, labels=range(1, 6), label_column='stars')
            splits[split].append(_ds)
    lang_zh = ds.filter(lambda x: x['language']=='zh')
    _ds = option_sampling(lang_zh['train'], num_sample=8000/5, labels=range(1, 6), label_column='stars')
    train_ds, valid_ds = train_test_sampling(_ds, num_train=6000/5, labels=range(1, 6), label_column='stars')
    valid_ds, test_ds = train_test_sampling(valid_ds, num_train=1000/5, labels=range(1, 6), label_column='stars')
    splits['train'].append(train_ds)
    splits['validation'].append(valid_ds)
    splits['test'].append(test_ds)

    def arrange_column(ds):
        ds = ds.rename_column('stars', 'label')
        ds = ds.rename_column('review_body', 'review')
        ds = ds.rename_column('language', 'lang')
        ds = ds.remove_columns([col for col in ds.column_names if col not in ['review', 'label', 'lang']])
        return ds

    train_ds = arrange_column(datasets.concatenate_datasets(splits['train']))
    valid_ds = arrange_column(datasets.concatenate_datasets(splits['validation']))
    test_ds = arrange_column(datasets.concatenate_datasets(splits['test']))
    
    ordered_columns = ['review', 'label', 'lang']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_lti(data_dir):
    langs = ['eng', 'fra', 'deu', 'cmn', 'spa']
    langs32 = {
        'eng': 'en', 
        'fra': 'fr', 
        'deu': 'de', 
        'cmn': 'zh', 
        'spa': 'es'
    }
    ds = datasets.load_dataset('SEACrowd/lti_langid_corpus', trust_remote_code=True)
    ds = ds.rename_column('language', 'lang')
    ds = ds.filter(lambda x: x['lang'] in langs and 'strong' not in x['text'])
    
    def process_lang(item):
        item['lang'] = langs32[item['lang']]
        return item
    
    train_ds = option_sampling(ds['train'], num_sample=7000, labels=langs, label_column='lang')
    train_ds, valid_ds = train_test_sampling(train_ds, num_train=6000, labels=langs, label_column='lang')
    # valid_ds = option_sampling(ds['validation'], num_sample=1000, labels=langs, label_column='lang')
    test_ds = option_sampling(ds['test'], num_sample=1000, labels=langs, label_column='lang')
    
    train_ds = train_ds.map(process_lang)
    valid_ds = valid_ds.map(process_lang)
    test_ds = test_ds.map(process_lang)

    ordered_columns = ['text', 'lang']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_mlama(data_dir):
    ds = datasets.load_dataset('cis-lmu/m_lama', trust_remote_code=True)
    ds = ds.rename_column('language', 'lang')

    langs = ['en', 'fr', 'de', 'es', 'zh']
    ds = ds.filter(lambda x: x['lang'] in langs)
    df = ds['test'].to_pandas()
    uuids = set()
    for uuid, sdf in df.groupby('uuid'):
        if len(sdf) == 5:
            uuids.add(uuid)
    df = df[df['uuid'].isin(uuids)]
    rel2obj, subrel2obj, lang2obj2label = {}, {}, {'en': {}, 'fr': {}, 'de': {}, 'es': {}, 'zh': {}}
    for lang, rel, sub, obj, obj_label in zip(df['lang'], df['predicate_id'], df['sub_uri'], df['obj_uri'], df['obj_label']):
        subrel2obj.setdefault(f'{rel}-{sub}', set()).add(obj)
        if rel not in rel2obj:
            rel2obj[rel] = {}
        if obj not in rel2obj[rel]:
            rel2obj[rel][obj] = 1
        else:
            rel2obj[rel][obj] += 1
        lang2obj2label[lang][obj] = obj_label

    sample_uuids = random.sample(list(uuids), k=int(8000/5))
    train_uuids = random.sample(sample_uuids, k=int(6000/5))
    valid_uuids = random.sample(list(set(sample_uuids) - set(train_uuids)), k=int(1000/5))
    test_uuids = random.sample(list(set(sample_uuids) - set(train_uuids) - set(valid_uuids)), k=int(1000/5))

    def process_df(df):
        df['index'] = df.index
        return Dataset.from_pandas(df)

    train_ds = process_df(df[df['uuid'].isin(train_uuids)].reset_index())
    valid_ds = process_df(df[df['uuid'].isin(valid_uuids)].reset_index())
    test_ds = process_df(df[df['uuid'].isin(test_uuids)].reset_index())

    def create_label(item):
        cands = list(rel2obj[item['predicate_id']].keys())
        answers = subrel2obj[f"{item['predicate_id']}-{item['sub_uri']}"]
        non_ans = list(set(cands) - set(answers))
        non_ans_weights = [rel2obj[item['predicate_id']][obj] for obj in non_ans]
        options = weighted_sample(items=non_ans, weights=non_ans_weights, num_sample=3)
        assert len(set(options)) == 3
        options = [lang2obj2label[item['lang']][uri] for uri in options]
        answer_uri = random.sample(answers, 1)[0]
        answer = lang2obj2label[item['lang']][answer_uri]
        assert answer not in options, non_ans
        # options.append(answer)
        random.shuffle(options)
        assert len(set(options)) == 3, len(set(options))

        item['label'] = item['index'] % 4
        if item['index'] % 4 == 0:
            item['options'] = [lang2obj2label[item['lang']][answer_uri], options[0], options[1], options[2]]
        elif item['index'] % 4 == 1:
            item['options'] = [options[0], lang2obj2label[item['lang']][answer_uri], options[1], options[2]]
        elif item['index'] % 4 == 2:
            item['options'] = [options[0], options[1], lang2obj2label[item['lang']][answer_uri], options[2]]
        elif item['index'] % 4 == 3:
            item['options'] = [options[0], options[1], options[2], lang2obj2label[item['lang']][answer_uri]]
        item['text'] = item['template'].replace('[X]', item['sub_label'])
        return item

    removable_cols = [col for col in train_ds.column_names if col not in ['lang', 'uuid', 'text', 'options', 'label']]
    train_ds = train_ds.map(create_label).remove_columns(removable_cols)
    valid_ds = valid_ds.map(create_label).remove_columns(removable_cols)
    test_ds = test_ds.map(create_label).remove_columns(removable_cols)
    
    ordered_columns = ['uuid', 'text', 'options', 'label', 'lang']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_xnli(data_dir):
    langs = ['en', 'fr', 'de', 'es', 'zh']
    lang2ds = {}
    for lang in langs:
        ds = datasets.load_dataset('facebook/xnli', lang, trust_remote_code=True).shuffle(seed=42)
        if lang == 'en':
            train_ds = ds['train'].add_column("index", range(len(ds['train'])))
            valid_ds = ds['validation'].add_column("index", range(len(ds['validation'])))
            test_ds = ds['test'].add_column("index", range(len(ds['test'])))
            train_ds = option_sampling(train_ds, num_sample=int(6000/5), labels=range(3))
            valid_ds = option_sampling(valid_ds, num_sample=int(1000/5), labels=range(3))
            test_ds = option_sampling(test_ds, num_sample=int(1000/5), labels=range(3))
            train_idx = train_ds['index']
            valid_idx = valid_ds['index']
            test_idx = test_ds['index']
        else:
            train_ds = ds['train'].add_column("index", range(len(ds['train'])))
            valid_ds = ds['validation'].add_column("index", range(len(ds['validation'])))
            test_ds = ds['test'].add_column("index", range(len(ds['test'])))
            train_ds = train_ds.filter(lambda x: x['index'] in train_idx)
            valid_ds = valid_ds.filter(lambda x: x['index'] in valid_idx)
            test_ds = test_ds.filter(lambda x: x['index'] in test_idx)
        test_ds = test_ds.add_column("lang", [lang] * len(test_ds))
        valid_ds = valid_ds.add_column("lang", [lang] * len(valid_ds))
        train_ds = train_ds.add_column("lang", [lang] * len(train_ds))

        lang2ds[lang] = {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
    train_ds = datasets.concatenate_datasets([lang2ds[lang]['train'] for lang in langs])
    valid_ds = datasets.concatenate_datasets([lang2ds[lang]['valid'] for lang in langs])
    test_ds = datasets.concatenate_datasets([lang2ds[lang]['test'] for lang in langs])

    ordered_columns = [ 'index', 'premise', 'hypothesis', 'label', 'lang']
    # train_ds.to_csv(data_dir / "train.csv", index=False, columns=ordered_columns)
    # valid_ds.to_csv(data_dir / "valid.csv", index=False, columns=ordered_columns)
    # test_ds.to_csv(data_dir / "test.csv", index=False, columns=ordered_columns)
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_mpostag(data_dir):
    lang2urls = {
        'en': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/refs/heads/master/en_pud-ud-test.conllu',
        'de': 'https://raw.githubusercontent.com/UniversalDependencies/UD_German-PUD/refs/heads/master/de_pud-ud-test.conllu',
        'fr': 'https://raw.githubusercontent.com/UniversalDependencies/UD_French-PUD/refs/heads/master/fr_pud-ud-test.conllu', 
        'es': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-PUD/refs/heads/master/es_pud-ud-test.conllu',
        'zh': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-PUD/refs/heads/master/zh_pud-ud-test.conllu'
    }

    lang2items = {}
    lang2tag_distri = {}
    for lang, url in lang2urls.items():
        lang2tag_distri[lang] = {}
        lang2items[lang] = []
        data = urlopen(url).read().decode('utf-8')
        sentences = parse(data)
        for sentence in sentences:
            for token_info in sentence:
                if token_info['upos'] not in lang2tag_distri[lang]:
                    lang2tag_distri[lang][token_info['upos']] = 1
                else:
                    lang2tag_distri[lang][token_info['upos']] += 1
    for lang, url in lang2urls.items():
        lang2items[lang] = []
        data = urlopen(url).read().decode('utf-8')
        all_tags = list(set(lang2tag_distri[lang].keys()).intersection(set(universal_pos_tags)))
        sentences = parse(data)
        for sentence in sentences:
            token_cnt = Counter([token_info['form'] for token_info in sentence if token_info['upos']!='PUNCT'])
            cand_tokens = [token for token in token_cnt if token_cnt[token] == 1]
            all_tokens = [token['form'] for token in sentence]
            items = []
            for token_info in sentence:
                if token_info['form'] not in cand_tokens:
                    continue
                non_tags = list(set(all_tags) - set([token_info['upos']]))
                non_tags_weights = [lang2tag_distri[lang][tag] for tag in non_tags]
                options = weighted_sample(items=non_tags, weights=non_tags_weights, num_sample=3)
                assert len(set(options)) == 3
                options.append(token_info['upos'])
                random.shuffle(options)
                assert len(set(options)) == 4
                items.append((sentence.metadata['text'], all_tokens, token_info['form'], options, options.index(token_info['upos']), lang))
            lang2items[lang].extend(items)
    trains, valids, tests = [], [], []
    for lang, items in lang2items.items():
        texts, all_tokens, tokens, options, labels, langs = zip(*items)
        ds = Dataset.from_pandas(pd.DataFrame({
            'text': texts,
            'tokens': all_tokens, 
            'target_token': tokens, 
            'options': options,
            'label': labels,
            'lang': langs
        }))
        ds = option_sampling(ds, num_sample=int(8000/5), labels=range(4))
        train_ds, valid_ds = train_test_sampling(ds, int(6000/5), labels=range(4))
        test_ds, valid_ds = train_test_sampling(valid_ds, int(1000/5), labels=range(4))
        trains.append(train_ds)
        valids.append(valid_ds)
        tests.append(test_ds)
    train_ds = datasets.concatenate_datasets(trains)
    valid_ds = datasets.concatenate_datasets(valids)
    test_ds = datasets.concatenate_datasets(tests)

    ordered_columns = ['text', 'tokens', 'target_token', 'options', 'label', 'lang']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)
    
def process_clang8(data_dir):
    data_path = "../raw_data/clang8/clang8_source_target_en.spacy_tokenized.tsv"
    df = pd.read_csv(data_path, sep='\t', header=None, on_bad_lines='skip', names=["false", "true"])
    df = df[df["false"].str.strip().str.lower()!=df["true"].str.strip().str.lower()]
    df = df[df["false"].str.len()>50]
    
    import numpy as np
    def create_items_vectorized(df):
        n = len(df)
        idx = np.random.randint(0, 2, size=n)
        sentence1 = np.where(idx == 0, df['true'], df['false'])
        sentence2 = np.where(idx == 1, df['true'], df['false'])

        df['sentence1'] = sentence1
        df['sentence2'] = sentence2
        df['acceptable'] = idx
        df = df.drop(columns=['false', 'true'])
        return df

    ndf = create_items_vectorized(df)
    ds = Dataset.from_pandas(ndf)
    ds = option_sampling(ds, num_sample=8000, labels=range(2), label_column='acceptable')
    train_ds, valid_ds = train_test_sampling(ds, 6000, labels=range(2), label_column='acceptable')
    test_ds, valid_ds = train_test_sampling(valid_ds, 1000, labels=range(2), label_column='acceptable')

    ordered_columns = ['sentence1', 'sentence2', 'acceptable']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)

def process_postag(data_dir):
    urls = {
        'train': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-train.conllu',
        'validation': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-dev.conllu',
        'test': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-test.conllu'
    }
    split2ds = {}
    for split, url in urls.items():
        all_items = []
        data = urlopen(url).read().decode('utf-8')
        sentences = parse(data)

        tag_distri = {}
        for sentence in sentences:
            for token_info in sentence:
                if token_info['upos'] not in tag_distri:
                    tag_distri[token_info['upos']] = 1
                else:
                    tag_distri[token_info['upos']] += 1

        all_tags = list(set(tag_distri.keys()).intersection(set(universal_pos_tags)))
        for sentence in sentences:
            token_cnt = Counter([token_info['form'] for token_info in sentence if token_info['upos']!='PUNCT'])
            cand_tokens = [token for token in token_cnt if token_cnt[token] == 1]
            all_tokens = [token['form'] for token in sentence]
            items = []
            for token_info in sentence:
                if token_info['form'] not in cand_tokens:
                    continue
                
                non_tags = list(set(all_tags) - set([token_info['upos']]))
                non_tags_weights = [tag_distri[tag] for tag in non_tags]
                options = weighted_sample(items=non_tags, weights=non_tags_weights, num_sample=3)
                options.append(token_info['upos'])
                random.shuffle(options)
                assert len(options) == 4
                items.append((
                    sentence.metadata['text'], 
                    all_tokens, 
                    token_info['form'], 
                    options, 
                    options.index(token_info['upos'])))
            all_items.extend(items)

        texts, all_tokens, tokens, options, labels = zip(*all_items)
        ds = Dataset.from_pandas(pd.DataFrame({
            'text': texts,
            'tokens': all_tokens, 
            'target_token': tokens, 
            'options': options,
            'label': labels
        }))
        num_sample = 6000 if split == 'train' else 1000
        split2ds[split] = option_sampling(ds, num_sample=num_sample, labels=range(4))
    
    ordered_columns = ['text', 'tokens', 'target_token', 'options', 'label']
    dump_datasets(data_dir, split2ds['train'], split2ds['validation'], split2ds['test'], ordered_columns)
    
def process_chunking(data_dir):
    dataset = load_dataset("eriktks/conll2000")
    chunk_label_mapping = dataset["train"].features["chunk_tags"].feature.names
    
    def extract_chunks(tokens, chunk_tags):
        chunks = []
        current_chunk = []
        current_chunk_type = None

        for token, tag_id in zip(tokens, chunk_tags):
            tag = chunk_label_mapping[tag_id]    
            if tag == "O":  # Outside any chunk
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                    current_chunk = []
                    current_chunk_type = None
            elif tag.startswith("B-"):  # Beginning of a new chunk
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                current_chunk_type = tag[2:]  # Get the chunk type (e.g., "NP")
                current_chunk = [token]
            elif tag.startswith("I-") and current_chunk_type == tag[2:]:  # Continuation of the same chunk
                current_chunk.append(token)
            else:  # Handle invalid transitions (optional)
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                current_chunk_type = tag[2:]
                current_chunk = [token]

        if current_chunk:
            chunks.append((current_chunk_type, current_chunk))
        
        return chunks
    
    all_tags = set([tag.split('-')[1] for tag in chunk_label_mapping if tag != "O"])
    tag2cnt = {}
    for sentence in dataset["train"]:
        tokens = sentence["tokens"]
        chunk_tags = sentence["chunk_tags"]
        for chunk_tag, chunk in extract_chunks(tokens, chunk_tags):
            if chunk_tag not in tag2cnt:
                tag2cnt[chunk_tag] = 1
            else:
                tag2cnt[chunk_tag] += 1
    all_tags = all_tags.intersection(set(tag2cnt.keys()))
    
    texts, all_chunks, target_chunk, options, labels = [], [], [], [], []
    for sentence in dataset["train"]:
        tokens = sentence["tokens"]
        chunk_tags = sentence["chunk_tags"]
        chunks = [chunk for chunk_tag, chunk in extract_chunks(tokens, chunk_tags)]
        for chunk_tag, chunk in extract_chunks(tokens, chunk_tags):
            texts.append(' '.join(tokens))
            all_chunks.append(chunks)
            target_chunk.append(chunk)
            
            non_tags = list(set(all_tags) - set([chunk_tag]))
            non_tags_weights = [tag2cnt[tag] for tag in non_tags]
            _options = weighted_sample(items=non_tags, weights=non_tags_weights, num_sample=3)
            _options.append(chunk_tag)
            random.shuffle(_options)
            options.append(_options)
            assert len(set(_options)) == 4
            labels.append(_options.index(chunk_tag))

    ds = Dataset.from_pandas(pd.DataFrame({
        'text': texts,
        'chunks': all_chunks, 
        'target_chunk': target_chunk, 
        'options': options,
        'label': labels
    }))
            
    ds = option_sampling(ds, num_sample=8000, labels=range(4))
    train_ds, test_ds = train_test_sampling(ds, 6000, range(4))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, range(4))
    ordered_columns = ['text', 'chunks', 'target_chunk', 'options', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


def process_ner(data_dir):
    dataset = load_dataset("eriktks/conll2003")
    dataset = dataset.filter(lambda x: len(x['tokens'])>10)
    chunk_label_mapping = dataset["train"].features["ner_tags"].feature.names

    def extract_chunks(tokens, ner_tags):
        chunks = []
        current_chunk = []
        current_chunk_type = None

        for token, tag_id in zip(tokens, ner_tags):
            tag = chunk_label_mapping[tag_id]    
            if tag == "O":  # Outside any chunk
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                    current_chunk = []
                    current_chunk_type = None
            elif tag.startswith("B-"):  # Beginning of a new chunk
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                current_chunk_type = tag[2:]  # Get the chunk type (e.g., "NP")
                current_chunk = [token]
            elif tag.startswith("I-") and current_chunk_type == tag[2:]:  # Continuation of the same chunk
                current_chunk.append(token)
            else:  # Handle invalid transitions (optional)
                if current_chunk:
                    chunks.append((current_chunk_type, current_chunk))
                current_chunk_type = tag[2:]
                current_chunk = [token]

        if current_chunk:
            chunks.append((current_chunk_type, current_chunk))
        
        return chunks

    all_tags = set([tag.split('-')[1] for tag in chunk_label_mapping if tag != "O"])
    tag2cnt = {}
    for sentence in dataset["train"]:
        tokens = sentence["tokens"]
        ner_tags = sentence["ner_tags"]
        for chunk_tag, chunk in extract_chunks(tokens, ner_tags):
            if chunk_tag not in tag2cnt:
                tag2cnt[chunk_tag] = 1
            else:
                tag2cnt[chunk_tag] += 1
    all_tags = all_tags.intersection(set(tag2cnt.keys()))

    texts, all_chunks, target_chunk, options, labels = [], [], [], [], []
    for sentence in dataset["train"]:
        tokens = sentence["tokens"]
        ner_tags = sentence["ner_tags"]
        chunks = [chunk for chunk_tag, chunk in extract_chunks(tokens, ner_tags)]
        for chunk_tag, chunk in extract_chunks(tokens, ner_tags):
            texts.append(' '.join(tokens))
            all_chunks.append(chunks)
            target_chunk.append(chunk)
            
            non_tags = list(set(all_tags) - set([chunk_tag]))
            non_tags_weights = [tag2cnt[tag] for tag in non_tags]
            _options = weighted_sample(items=non_tags, weights=non_tags_weights, num_sample=3)
            _options.append(chunk_tag)
            random.shuffle(_options)
            options.append(_options)
            assert len(set(_options)) == 4
            labels.append(_options.index(chunk_tag))

    ds = Dataset.from_pandas(pd.DataFrame({
        'text': texts,
        'chunks': all_chunks, 
        'target_chunk': target_chunk, 
        'options': options,
        'label': labels
    }))
            
    ds = option_sampling(ds, num_sample=8000, labels=range(4))
    train_ds, test_ds = train_test_sampling(ds, 6000, range(4))
    valid_ds, test_ds = train_test_sampling(test_ds, 1000, range(4))
    ordered_columns = ['text', 'chunks', 'target_chunk', 'options', 'label']
    dump_datasets(data_dir, train_ds, valid_ds, test_ds, ordered_columns)


data_types = {
    "linguistic": {
        "clang8": process_clang8,
        "postag": process_postag,
        "chunking": process_chunking,
        "ner": process_ner,
    },
    "classification": {
        "agnews": process_agnews, 
        "amazon-reviews": process_amazonreview, 
        "imdb": process_imdb}, 
    "fact": {
        "myriadlama": process_myriadlama, 
        "fever": process_fever, 
        "commensenseqa": process_commensenseqa, 
        "templama": process_templama
        },
    "nli": {
        "paws": process_paws, 
        "mnli": process_mnli, 
        "swag": process_swag, 
    },
    "self-reflection": {
        "halueval": process_halueval, 
        "toxicity": process_toxicity, 
        "stereoset": process_stereoset,
    },
    "multilingual": {
        "amazon-review-multi": process_multiamazon, 
        "lti": process_lti, 
        "mlama": process_mlama,
        'xnli': process_xnli,
        'mpostag': process_mpostag
    }
}

if __name__ == '__main__':
    for genre in data_types:
        os.makedirs((BASE_DIR / genre).as_posix(), exist_ok=True)
        for task in data_types[genre]:
            os.makedirs((BASE_DIR / genre / task).as_posix(), exist_ok=True)
            print(f"==== Downloading and processing Task:{task} of Genre:{genre} ====")
            data_dir = BASE_DIR / genre / task
            process_func = data_types[genre][task]
            json_files = [data_dir/'train.json', data_dir/'valid.json', data_dir/'test.json']
            if not all([json_fn.is_file() and json_fn.exists() for json_fn in json_files]):
                process_func(data_dir)
            else:
                print(f"Dataset for {task} are already downloaded and processed in {data_dir}")

            train_ds = Dataset.from_json((data_dir/'train.json').as_posix())
            valid_ds = Dataset.from_json((data_dir/'valid.json').as_posix())
            test_ds = Dataset.from_json((data_dir/'test.json').as_posix())

            if task == 'stereoset' or task == 'clang8':
                continue
            assert len(train_ds)==6000, len(train_ds)
            assert len(valid_ds)==1000, len(valid_ds)
            assert len(test_ds)==1000, len(test_ds)

            if 'label' not in train_ds.column_names:
                if task == 'lti':
                    assert len(Counter(train_ds['lang'])) == 5, len(Counter(train_ds['lang'])) 
                    assert len(Counter(valid_ds['lang'])) == 5, len(Counter(valid_ds['lang'])) 
                    assert len(Counter(test_ds['lang'])) == 5, len(Counter(test_ds['lang'])) 
                elif task == 'halueval':
                    assert len(Counter(train_ds['hallucination'])) == 2, len(Counter(train_ds['hallucination'])) 
                    assert len(Counter(valid_ds['hallucination'])) == 2, len(Counter(valid_ds['hallucination'])) 
                    assert len(Counter(test_ds['hallucination'])) == 2, len(Counter(test_ds['hallucination'])) 
                elif task == 'toxicity':
                    assert len(Counter(train_ds['toxic'])) == 2, len(Counter(train_ds['toxic'])) 
                    assert len(Counter(valid_ds['toxic'])) == 2, len(Counter(valid_ds['toxic'])) 
                    assert len(Counter(test_ds['toxic'])) == 2, len(Counter(test_ds['toxic'])) 
                else:
                    raise ValueError()
                continue 
            
            if len(set(train_ds['label'])) == 3:
                continue
            assert len(set(Counter(train_ds['label']).values())) == 1, len(set(Counter(train_ds['label']).values())) 
            assert len(set(Counter(valid_ds['label']).values())) == 1, len(set(Counter(valid_ds['label']).values())) 
            assert len(set(Counter(test_ds['label']).values())) == 1, len(set(Counter(test_ds['label']).values())) 

            if genre == 'multilingual':
                assert len(set(Counter(train_ds['lang']).values())) == 1, len(set(Counter(train_ds['lang']).values())) 
                assert len(set(Counter(valid_ds['lang']).values())) == 1, len(set(Counter(valid_ds['lang']).values())) 
                assert len(set(Counter(test_ds['lang']).values())) == 1, len(set(Counter(test_ds['lang']).values()))            
