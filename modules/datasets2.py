import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        #unlikelihood = [i for i in self.tokenizer.freq_word_order if self.tokenizer.freq_word_order[i]>=100]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            '''
            unlikelihood_word = ' '.join([j for j in unlikelihood if j not in self.examples[i]['report']])
            unlikelihood_word = tokenizer(unlikelihood_word)
            a = torch.sum(F.one_hot(torch.tensor(unlikelihood_word),len(self.tokenizer.token2idx)+1),dim=0)
            a[0] = 1
            a = a.unsqueeze(0)
            self.examples[i]['negative'] = a
            '''
            self.examples[i]['negative']='unk'


    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        neg = example['negative']
        seq_length = len(report_ids)
        sample1 = (image_id, image, report_ids, report_masks, seq_length,neg)

        try:
            example = self.examples[idx+1]
        except:
            example = self.examples[idx-1]

        image_id2 = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image2 = torch.stack((image_1, image_2), 0)
        report_ids2 = example['ids']
        report_masks2 = example['mask']
        neg2 = example['negative']
        seq_length2 = len(report_ids2)
        sample2 = (image_id2, image2, report_ids2, report_masks2, seq_length2,neg2)

        return sample1,sample2


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        neg = example['negative']
        sample1 = (image_id, image, report_ids, report_masks, seq_length, neg)


        try:
            example = self.examples[idx+1]
        except:
            example = self.examples[idx-1]

        image_id2 = example['id']
        image_path = example['image_path']
        image2 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image2 = self.transform(image2)
        report_ids2 = example['ids']
        report_masks2 = example['mask']
        neg2 = example['negative']
        seq_length2 = len(report_ids2)
        sample2 = (image_id2, image2, report_ids2, report_masks2, seq_length2,neg2)

        return sample1,sample2
