import logging
import os
from abc import abstractmethod
import numpy as np
import torch
from numpy import inf
from model.models import MetaLearning
import torch.nn.functional as F

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.metaRL = MetaLearning(model.tokenizer).to(self.device)
        self.metarl_opt = torch.optim.Adam(self.metaRL.parameters(), lr=0.001, betas=(0.9, 0.99), eps=0.0000001)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        #if args.resume is not None:
            #args.resume = '/home/ywu10/Documents/R2GenCMN/results/iu_xray/model_best.pth'
        self._resume_checkpoint(args.resume)


    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))
            #break
            '''
            if self.best_recorder['test']['test_BLEU_1']>log['test_BLEU_1']:
                self._save_checkpoint(epoch, save_best=True)
            else:
                self._save_checkpoint(epoch, save_best=False)
            '''
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break
                    
            #if epoch % self.save_period == 0:
            #    self._save_checkpoint(epoch, save_best=best)


    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        #resume_path = '/home/ywu10/Documents/R2GenCMN/models/model_iu_xray.pth'
        #resume_path = '/home/ywu10/Documents/R2GenCMN/models/model_mimic_cxr.pth'
        resume_path = '/home/ywu10/Documents/R2GenCMN/results/iu_xray/best1.pth'
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        train_loss = 0

        self.model.train()
        count = 0
        for batch_idx, (images_id, images, reports_ids, reports_masks,_, \
                        images_id2, images2, reports_ids2, reports_masks2,_) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)

            images2, reports_ids2, reports_masks2 = images2.to(self.device), reports_ids2.to(self.device), reports_masks2.to(
                self.device)

            origin_param = self.model.state_dict()
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            lm_param = self.model.state_dict()


            #optimize meta-RL
            self.optimizer.zero_grad()
            #self.metarl_opt.zero_grad()
            self.model.load_state_dict(origin_param)
            output = torch.mean(output,dim=1)
            reports_ids_ = (torch.sum(F.one_hot(reports_ids,len(self.model.tokenizer.idx2token)+1),dim=1) > 0).long()
            action,entropy = self.metaRL.predict_action(output*reports_ids_)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks) + self.unlikelihood_loss(output,action,reports_masks)

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            rl_param = self.model.state_dict()

            #optimizing rl by calculating reward on test set
            with torch.no_grad():
                self.model.load_state_dict(lm_param)
                output,_ = self.model(images2, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids2[:, 1:].cpu().numpy())
                lm_reward = self.reward(ground_truths,reports)

                self.model.load_state_dict(rl_param)
                output,_ = self.model(images2, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids2[:, 1:].cpu().numpy())
                rl_reward = torch.tensor(self.reward(ground_truths,reports)).cuda(0)

            self.model.load_state_dict(origin_param)
            self.metarl_opt.zero_grad()
            output = self.model(images, reports_ids, mode='train')
            output = torch.mean(output,dim=1)
            reports_ids_ = (torch.sum(F.one_hot(reports_ids,len(self.model.tokenizer.idx2token)+1),dim=1) > 0).long()
            action,entropy = self.metaRL.predict_action(output*reports_ids_)
            predict_reward = self.metaRL.predict_reward(output*reports_ids_,action)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks) + self.unlikelihood_loss(output,action,reports_masks)

            reward = rl_reward - lm_reward
            loss = (reward-predict_reward)**2 - torch.mean((reward-predict_reward) * loss, dim=0) - 0.1*torch.mean(entropy)
            loss.backward()
            self.metarl_opt.step()

            if reward>=0:
                self.model.load_state_dict(rl_param)
                count += 1
            else:
                self.model.load_state_dict(lm_param)

        print(f'rl train:{count/len(self.train_dataloader)}')

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks,_, \
                            images_id2, images2, reports_ids2, reports_masks2,_) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output,_ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks,_, \
                            images_id2, images2, reports_ids2, reports_masks2,_) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output,_ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            self.imbalanced_eval(test_res,test_gts)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        print(f'predict:{test_res[-1]}')
        print(f'ground_truth:{test_gts[-1]}')

        return log

    def reward(self,tgt,pre):
        words = [w for w in self.model.tokenizer.token2idx][:-2]
        recall_ = []
        precision_ = []
        right_ = []
        gap = len(words)//8
        for index in range(0,len(words)-gap,gap):
            right = 0
            recall = 0
            precision = 0
            for i in range(len(tgt)):
                a = [j for j in tgt[i].split() if j in words[index:index+gap]]
                b = [j for j in pre[i].split() if j in  words[index:index+gap]]
                right += len([j for j in a if j in b])
                recall += len(a)
                precision += len(b)
            recall_.append(recall)
            precision_.append(precision)
            right_.append(right)
        recall = np.array(right_)/np.array(recall_)
        precision = np.array(right_)/np.array(precision_)
        score = 2 * precision * recall / (precision+recall)
        return np.sum(np.nan_to_num(score))

    def imbalanced_eval(self,pre,tgt,n):

        #words = dict(sorted(dict(self.model.tokenizer.counter).items(), key=lambda x: x[1]))
        words = [w for w in self.model.tokenizer.token2idx][:-2]
        recall_ = []
        precision_ = []
        right_ = []
        gap = len(words)//n
        for index in range(0,len(words)-gap,gap):
            right = 0
            recall = 0
            precision = 0
            for i in range(len(tgt)):
                a = [j for j in tgt[i].split() if j in words[index:index+gap]]
                b = [j for j in pre[i].split() if j in  words[index:index+gap]]
                right += len([j for j in a if j in b])
                recall += len(a)
                precision += len(b)
            recall_.append(recall)
            precision_.append(precision)
            right_.append(right)
        print(f'recall:{np.array(right_)/np.array(recall_)}')
        print(f'precision:{np.array(right_)/np.array(precision_)}')
        print(precision_)
        print(recall_)

    def unlikelihood_loss(self,output,negative_word,mask):

        output = torch.sum(mask[:,1:].unsqueeze(-1)*output,dim=1)/torch.sum(mask)
        loss = -torch.log(torch.clamp(1.0 - (output * negative_word).exp(), min=1e-20))
        return torch.mean(torch.mean(loss,dim=-1),dim=-1)

    def _train_epoch2(self, epoch):

        train_loss = 0
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #self.imbalanced_eval(val_res,val_gts)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))

        self.model.eval()
        with torch.no_grad():

            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            #test_gts, test_res = self.heihei()
            self.imbalanced_eval(test_res,test_gts)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            print(test_met)
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
