import os
import numpy as np
import pickle
from tqdm import trange

from landmark.umeyama import umeyama


class Candidates:

    def __init__(self, ldm_all_path: str, skull_candidates_dir: str) -> None:
        
        self.ldm_all = np.load(ldm_all_path)
        self.face_num, self.ldm_num, self.coord_num = self.ldm_all.shape
        self.skull_candidates_dir = skull_candidates_dir

    def match(self, skull_ldm: np.array, skull_id: str, candidate_rank: list):
        
        candidate_path = os.path.join(self.skull_candidates_dir, f'{skull_id}.pkl')
        face_meta_list = []
        # if os.path.exists(candidate_path):
        if False:

            with open(candidate_path, 'rb') as f:
                face_meta_list = pickle.load(f)

            # print(face_meta_list)

        else:
            for i in trange(self.face_num):
                # c, R, t = umeyama(skull_ldm.T, self.ldm_all[i].T)
                # t = t.T

                c = 0.133
                R = np.array([
                    [-1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ])
                t = np.array([-0.0020, 0.005, -0.022])
                
                skull_ldm_realigned = c * skull_ldm @ R + t

                diff = skull_ldm_realigned - self.ldm_all[i]
                err = np.linalg.norm(diff, axis=-1).mean()
                # print(c, R, t, err)

                face_meta_list.append([i, c, R, t, err])
                # if i > 10:
                #     break
            
            face_meta_list.sort(key=lambda x: x[-1])

            saved_face_meta_list = []
            for i in range(len(candidate_rank)):
                saved_face_meta_list.append([candidate_rank[i]] + face_meta_list[candidate_rank[i]])
            
            print(saved_face_meta_list)
            with open(candidate_path, 'wb') as f:
                pickle.dump(saved_face_meta_list, f)
