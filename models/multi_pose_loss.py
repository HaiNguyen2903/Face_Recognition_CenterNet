from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss

from models.oracle_utils import gen_oracle_map
from config import *

class MultiPoseLoss(torch.nn.Module):
    def __init__(self):
        super(MultiPoseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = FocalLoss()
        self.crit_kp = RegWeightedL1Loss()
        self.crit_reg = RegL1Loss() 

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        lm_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0

        for s in range(NUM_STACKS):
            output = outputs[s]
            output['hm'] = output['hm']
        # if opt.hm_hp and not opt.mse_loss:
        #   output['hm_hp'] = _sigmoid(output['hm_hp'])
        
            if EVAL_ORACLE_KPS_HM:
                output['hm_hp'] = batch['hm_hp']
            if EVAL_ORACLE_CENTER_HM:
                output['hm'] = batch['hm']
            if EVAL_ORACLE_KPS:
                if DENSE_HP:
                    output['hps'] = batch['dense_hps']
                else:
                    output['hps'] = torch.from_numpy(gen_oracle_map(
                    batch['hps'].detach().cpu().numpy(), 
                    batch['ind'].detach().cpu().numpy(), 
                    OUTPUT_SIZE, OUTPUT_SIZE)).to(DEVICE)

            if EVAL_ORACLE_KPS_OFFSET:
                output['hp_offset'] = torch.from_numpy(gen_oracle_map(
                batch['hp_offset'].detach().cpu().numpy(), 
                batch['hp_ind'].detach().cpu().numpy(), 
                OUTPUT_SIZE, OUTPUT_SIZE)).to(DEVICE)


            hm_loss += self.crit(output['hm'], batch['hm']) / NUM_STACKS          # 1. focal loss, find the center of the target,
            
            # The loss of face bbox height and width
            if WH_WEIGHT > 0:
            #     wh_loss += (self.crit_reg(output['wh'], batch['reg_mask'],               
            #                             batch['ind'], batch['wh'], batch['wight_mask'])) 
                wh_loss += (self.crit_reg(output['wh'], batch['reg_mask'],               
                                        batch['ind'], batch['wh'])) / NUM_STACKS 

            # Down-sampling the center point of the face bbox, the required deviation compensation
            if REG_OFFSET and OFF_WEIGHT > 0:
                # off_loss += (self.crit_reg(output['hm_offset'], batch['reg_mask'],             
                #                         batch['ind'], batch['hm_offset'], batch['wight_mask'])) / NUM_STACKS
                off_loss += (self.crit_reg(output['hm_offset'], batch['reg_mask'],             
                                        batch['ind'], batch['hm_offset'])) / NUM_STACKS

            if DENSE_HP:
                mask_weight = batch['dense_hps_mask'].sum() + 1e-4
                lm_loss += ((self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                        batch['dense_hps'] * batch['dense_hps_mask']) / 
                                        mask_weight)) / NUM_STACKS
            else:
                lm_loss += (self.crit_kp(output['landmarks'], batch['hps_mask'],               # 4. Offset of the key point
                                        batch['ind'], batch['landmarks'])) / NUM_STACKS

        # if opt.reg_hp_offset and opt.off_weight > 0:                              # Center offset of the node
        #   hp_offset_loss += self.crit_reg(
        #     output['hp_offset'], batch['hp_mask'],
        #     batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        # if opt.hm_hp and opt.hm_hp_weight > 0:                                    #  Heat map of the nodes
        #   hm_hp_loss += self.crit_hm_hp(
        #     output['hm_hp'], batch['hm_hp']) / opt.num_stacks

        loss = HM_WEIGHT * hm_loss + WH_WEIGHT * wh_loss + \
            OFF_WEIGHT * off_loss + LM_WEIGHT * lm_loss
        
        # loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'lm_loss': lm_loss,
        #               'wh_loss': wh_loss, 'off_loss': off_loss}
        # return loss, loss_stats
        return loss