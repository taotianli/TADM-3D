import numpy as np
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms

import os
import pdb
import sys
import torch
from os.path import join

class SRDataSet(Dataset):
    def __init__(self, data, prefix='train', transforms=None):
        self.data_dir = join(data, prefix)
        self.prefix = prefix
        
        self.X = os.listdir(self.data_dir)

        self.len = len(self.X)
        
        #if self.task == 'srdiff':
        with open('configs/OASIS3_participant_data.txt') as f:
            lines = f.readlines()
            lines = [l.split("\t") for l in lines]
            
            lines = [l[:3] for l in lines]
            
            lines = np.array(lines)[1:]
            
            id_patient = lines[:, 0]
            days_to_visit = lines[:, 1].astype(int)
            age_at_visit = lines[:, 2].astype(float)
            
            _, idxs = np.unique(id_patient, return_index=True)

            self.id_patient  = id_patient[idxs]
            self.age_first_visit  = age_at_visit[idxs]
            self.day_first_visit  = days_to_visit[idxs]        

        #BUILDING DICT CONTAINING CONDITIONS DATA
        self.patients_conditions = {}
        with open('configs/OASIS3_patients_condition.txt') as f:
            lines = f.readlines()
            lines = [l.split("\t") for l in lines]
            lines = [l[:4] for l in lines]
            lines = np.array(lines)[1:]
            
            id_patient = lines[:, 0]
            days_to_visit = lines[:, 1].astype(int)
            age_at_visit = lines[:, 2].astype(float)
            condition = lines[:, 3].astype(int)
            
            self.patients_conditions['id_patient'] = id_patient
            self.patients_conditions['days_to_visit'] = days_to_visit
            self.patients_conditions['age_at_visit'] = age_at_visit
            self.patients_conditions['condition'] = condition
        
    def _get_item(self, index):
        return self.X[index]

    def __getitem__(self, index):
        pair = self._get_item(index)
        
        patient, mri_days, mri_days_next = pair.split("_")
        patient = "OAS" + patient
        mri_days = int(mri_days)
        mri_days_next = int(mri_days_next)
        mri_list = os.listdir(os.path.join(self.data_dir, pair))

        new_width = 144
        new_height = 120

        #COMPUTE AGES PER MRI
        age_first_visit = self.age_first_visit[self.id_patient == patient]
        day_first_visit = self.day_first_visit[self.id_patient == patient]
        
        #EXCLUDE NOT PRESENT PATIENT IN THE FILE PARTICIPANT_DATA.txt
        if len(age_first_visit) < 1:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)).float(), 'img_lr': torch.zeros((3,new_height,new_width)).float(), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0, 'split':self.prefix
            }

        mri_days = mri_days - day_first_visit
        mri_days_next = mri_days_next - day_first_visit

        if mri_days <= 0 or mri_days_next <= 0:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)), 'img_lr': torch.zeros((3,new_height,new_width)), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0, 'split':self.prefix
            }
      
        #COMPUTE DIFFERENT IN AGES
        #mri_diff_ages = ((1/365.0) * mri_days) #+ age_first_visit #YEAR BASED
        mri_diff_ages_next = ((1/30.0) * (mri_days_next - mri_days)) #+ age_first_visit #MONTH BASED
        age = ((1/365.0) * mri_days) + age_first_visit
        
        
        mri_list.sort()
        mri = mri_list[0]
        mri_next = mri_list[1]
        
        #READ CONDITION
        patient_conditions = self.patients_conditions['condition'][(self.patients_conditions['id_patient'] == patient)]
        if len(patient_conditions) > 0:
            patient_conditions = [patient_conditions[np.argmin(np.abs(np.array(patient_conditions)-mri_days[0]))]]
        
        if len(patient_conditions) < 1:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)).float(), 'img_lr': torch.zeros((3,new_height,new_width)).float(), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0, 'split':self.prefix
            }
        
        """data = nib.load(os.path.join(self.data_dir, pair, mri)).get_fdata()
        data_next = nib.load(os.path.join(self.data_dir, pair, mri_next)).get_fdata()

        #Select random slice
        img_hr = data_next[:, :, :]

        #Select random slice
        img_lr_up = data[:, :, :]
        
        img_hr = np.expand_dims(img_hr, axis=0)
        img_lr_up = np.expand_dims(img_lr_up, axis=0)"""
                 
        return {
            'img_hr': os.path.join(self.data_dir, pair, mri_next), 'img_lr': os.path.join(self.data_dir, pair, mri), 'item_name':(patient + "_" + str(mri_days)+"_"+str(mri_days_next)), 'diff_ages':mri_diff_ages_next[0], 'patient_condition':patient_conditions[0], 'age':age[0], 'split':self.prefix
        }

    def pre_process(self, img_hr):
        return img_hr

    def __len__(self):
        return self.len
