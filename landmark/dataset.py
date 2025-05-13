import numpy as np
import torch
from torch.utils import data


class DatasetLMK(data.Dataset):

    def __init__(
        self, 
        mode: str,
        dataset_cfg,
        # difficulty: float = 0
    ) -> None:
        super().__init__()

        self.lmk_all = np.load(dataset_cfg.ffhq_lmk_path)
        self.face_num, self.lmk_num, self.coord_num = self.lmk_all.shape

        self.lmk_dir_all = np.load(dataset_cfg.ffhq_lmk_dir_path)
        
        if mode == 'train':
            face_ids_path = dataset_cfg.train_list
        elif mode == 'val':
            face_ids_path = dataset_cfg.val_list
        else:
            raise NotImplementedError

        with open(face_ids_path, 'r') as f:
            lines = f.readlines()
        self.face_ids = [int(x.strip()) for x in lines]


    def __len__(self) -> int:
        return len(self.face_ids)

    def __getitem__(self, idx) -> dict:
        
        face_source_idx = self.face_ids[idx]
        lmk_source = self.lmk_all[face_source_idx]
        # lmk_dir = self.lmk_dir_all[idx]
        face_target_idx = np.random.choice(self.face_ids)
        while face_target_idx == face_source_idx:
            face_target_idx = np.random.choice(self.face_ids)
        lmk_target = self.lmk_all[face_target_idx]
        lmk_delta = lmk_target - lmk_source
        lmk_delta_input = np.zeros_like(lmk_delta)

        rand_lmk_idx = np.random.randint(0, self.lmk_num)
        lmk_delta_input[rand_lmk_idx] = lmk_delta[rand_lmk_idx]

        batch = {
            'id': (face_source_idx, face_target_idx),
            'lmk_source': torch.from_numpy(lmk_source).float(),
            'lmk_target': torch.from_numpy(lmk_target).float(),
            'lmk_delta_input': torch.from_numpy(lmk_delta_input).float()
        }

        return batch
