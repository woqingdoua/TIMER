import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

class MetaLearning(nn.Module):
    def __init__(self,tokenizer):
        super(MetaLearning, self).__init__()

        self.actor = nn.Linear(len(tokenizer.idx2token)+1,(len(tokenizer.idx2token)+1))
        self.critic = nn.Conv1d(2,1,10)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=0.002, betas=(0.9, 0.99), eps=0.0000001)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=0.002, betas=(0.9, 0.99), eps=0.0000001)

    def predict_action(self,loss):

        score = self.actor(loss)
        prob = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(score))
        samples = prob.sample()
        entropy = prob.entropy()

        return samples, entropy

    def predict_reward(self,loss,samples):
        input = torch.cat((loss.unsqueeze(1),samples.view(-1,1,loss.size()[-1])),dim=1)
        reward = self.critic(input).squeeze(1)
        reward = torch.mean(reward)
        return reward

    def learning(self,state,real_reward):

        self.critic_opt.zero_grad()
        action,_ = self.predict_action(state)
        pre_reward = self.predict_reward(state,action)
        loss = F.mse_loss(pre_reward,real_reward)
        loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        action,entropy = self.predict_action(state)
        pre_reward = self.predict_reward(state,action)
        loss = -pre_reward - 0.1*entropy
        loss.backward()
        self.actor_opt.step()
