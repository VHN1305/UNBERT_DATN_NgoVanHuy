import os
import pickle
import random
from typing import Dict, Any

import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')


# Xóa bỏ stopword, trả về chuỗi các từ không chứa stopword
def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])


# Lấy mẫu, số mẫu âm tính gấp 4 lần số mẫu dương. Trả về chuỗi các mẫu dưới dạng string
def sampling(imps, ratio=4):
    random.seed(42)
    pos = []
    neg = []
    for imp in imps.split():
        if imp[-1] == '1':
            pos.append(imp)
        else:
            neg.append(imp)
    n_neg = ratio * len(pos)
    if n_neg <= len(neg):
        neg = random.sample(neg, n_neg)
    else:
        neg = random.sample(neg * (n_neg // len(neg) + 1), n_neg)
    random.shuffle(neg)
    res = pos + neg
    random.shuffle(res)
    return ' '.join(res)


# Lớp MindDataset kế thừa từ lớp Dataset của PyTorch, dùng để tạo dữ liệu cho mô hình
class MindDataset(Dataset):
    # Hàm khởi tạo
    def __init__(self, root: str, tokenizer: AutoTokenizer,  # Tokenizer của BERT
                 mode: str = 'train',  # Chế độ train, dev hoặc test
                 split: str = 'small',  # Kích thước dữ liệu, small hoặc large
                 news_max_len: int = 20,  # Số từ tối đa của mỗi bài báo
                 hist_max_len: int = 20,  # Số bài báo tối đa trong lịch sử (news_history - user click history)
                 seq_max_len: int = 300  # Số từ tối đa của mỗi chuỗi
                 ) -> None:
        # Gọi hàm khởi tạo của lớp cha
        super(MindDataset, self).__init__()
        self.data_path = os.path.join(root, split)
        self._mode = mode
        self._split = split

        self._tokenizer = tokenizer
        self._mode = mode
        self._news_max_len = news_max_len
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len

        self._examples = self.get_examples(negative_sampling=4)
        # print(self._examples.head())
        self._news = self.process_news()

    # Hàm lấy dữ liệu từ file behaviors.tsv
    def get_examples(self, negative_sampling: bool = None  # Tỉ lệ negative sampling
                     ) -> Any:
        # Đọc dữ liệu từ file behaviors.tsv
        # Nếu split là small, file behaviors.tsv không có header, các trường được đọc theo thứ tự
        # Nếu split là large, file behaviors.tsv có header, các trường được đọc theo tên
        # Trả về DataFrame chứa dữ liệu

        behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        if self._split == 'small':
            df = pd.read_csv(behavior_file, sep='\t', header=None,
                             names=['user_id', 'time', 'news_history', 'impressions'])
            df['impression_id'] = list(range(len(df)))
        else:
            df = pd.read_csv(behavior_file, sep='\t', header=None,
                             names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        if self._mode == 'train':
            df = df.dropna(subset=['news_history'])
        df['news_history'] = df['news_history'].fillna('')

        if self._mode == 'train' and negative_sampling is not None:
            df['impressions'] = df['impressions'].apply(lambda x: sampling(x, ratio=negative_sampling))
        df = df.drop('impressions', axis=1).join(
            df['impressions'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('impression'))
        if self._mode == 'test':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
        else:
            df[['news_id', 'click']] = df['impression'].str.split('-', expand=True)
        df['click'] = df['click'].astype(int)
        return df

    def process_news(self) -> Dict[str, Any]:
        # Đọc thông tin bài báo từ file news_dict_bge.pkl
        # Nếu file tồn tại, đọc thông tin từ file
        filepath = os.path.join(self.data_path, 'news_dict.pkl')
        if os.path.exists(filepath):
            print('Loading news info from', filepath)
            with open(filepath, 'rb') as fin: news = pickle.load(fin)
            return news
        news = dict()
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        if self._split == 'large':
            news = self.read_news(news, os.path.join(self.data_path, 'test'))

        print('Saving news info from', filepath)
        with open(filepath, 'wb') as fout:
            pickle.dump(news, fout)
        return news

    def read_news(self, news: Dict[str, Any], filepath: str, drop_stopword: bool = True, ) -> Dict[str, Any]:
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            info = dict()
            splitted = line.strip('\n').split('\t')
            news_id = splitted[0]
            if news_id in news:
                continue
            title = splitted[3].lower()
            abstract = splitted[4].lower()
            if drop_stopword:
                title = remove_stopword(title)
                abstract = remove_stopword(abstract)
            news[news_id] = dict()
            title_words = self._tokenizer.tokenize(title)
            news[news_id]['title'] = self._tokenizer.convert_tokens_to_ids(title_words)
            abstract_words = self._tokenizer.tokenize(abstract)
            news[news_id]['abstract'] = self._tokenizer.convert_tokens_to_ids(abstract_words)
        return news

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        news_segment_ids = torch.tensor([item['news_segment_ids'] for item in batch])
        sentence_ids = torch.tensor([item['sentence_ids'] for item in batch])
        sentence_mask = torch.tensor([item['sentence_mask'] for item in batch])
        sentence_segment_ids = torch.tensor([item['sentence_segment_ids'] for item in batch])
        inputs = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                  'news_segment_ids': news_segment_ids, 'sentence_ids': sentence_ids, 'sentence_mask': sentence_mask,
                  'sentence_segment_ids': sentence_segment_ids, }
        if self._mode == 'train':
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['label'] = torch.tensor([item['label'] for item in batch])
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            return inputs
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def pack_bert_features(self, example: Any):
        curr_news = self._news[example['news_id']]['title'][:self._news_max_len]  # 1 candidate
        news_segment_ids = []
        hist_news = []
        sentence_ids = [0, 1, 2]
        for i, ns in enumerate(example['news_history'].split()[:self._hist_max_len]):
            ids = self._news[ns]['title'][:self._news_max_len]
            hist_news += ids
            news_segment_ids += [i + 2] * len(ids)
            sentence_ids.append(sentence_ids[-1] + 1)
        # Candidate - history-click
        tmp_hist_len = self._seq_max_len - len(curr_news) - 3
        hist_news = hist_news[:tmp_hist_len]
        input_ids = [self._tokenizer.cls_token_id] + curr_news + [self._tokenizer.sep_token_id] + hist_news + [
            self._tokenizer.sep_token_id]
        news_segment_ids = [0] + [1] * len(curr_news) + [0] + news_segment_ids[:tmp_hist_len] + [
            0]  # 0: curr_news, 1->end: hist_news
        segment_ids = [0] * (len(curr_news) + 2) + [1] * (len(hist_news) + 1)  # 0: curr_news, 1: hist_news
        input_mask = [1] * len(input_ids)  # attention_mask - 1 for input_ids, 0 for padding

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len
        news_segment_ids = news_segment_ids + [0] * padding_len

        sentence_segment_ids = [0] * 3 + [1] * (len(sentence_ids) - 3)  # 0: curr_news, 1: hist_news
        sentence_mask = [1] * len(sentence_ids)  # 1 for sentence_ids, 0 for padding

        sentence_max_len = 3 + self._hist_max_len
        sentence_mask = [1] * len(sentence_ids)
        padding_len = sentence_max_len - len(sentence_ids)
        sentence_ids = sentence_ids + [0] * padding_len  # Chuỗi tin tức???
        sentence_mask = sentence_mask + [0] * padding_len
        sentence_segment_ids = sentence_segment_ids + [0] * padding_len

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        assert len(news_segment_ids) == self._seq_max_len

        assert len(sentence_ids) == sentence_max_len
        assert len(sentence_mask) == sentence_max_len
        assert len(sentence_segment_ids) == sentence_max_len

        return input_ids, input_mask, segment_ids, news_segment_ids, sentence_ids, sentence_mask, sentence_segment_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples.iloc[index]
        input_ids, input_mask, segment_ids, news_segment_ids, sentence_ids, sentence_mask, sentence_segment_ids = self.pack_bert_features(
            example)
        inputs = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                  'news_segment_ids': news_segment_ids, 'sentence_ids': sentence_ids, 'sentence_mask': sentence_mask,
                  'sentence_segment_ids': sentence_segment_ids, }
        # print(len(inputs))
        if self._mode == 'train':
            inputs['label'] = example['click']
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = example['impression_id']
            inputs['label'] = example['click']
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = example['impression_id']
            return inputs
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return len(self._examples)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    train_set = MindDataset("..\data", tokenizer=tokenizer, mode='train', split='small', news_max_len=20,
        hist_max_len=20, seq_max_len=300)
    train_set.get_examples()
