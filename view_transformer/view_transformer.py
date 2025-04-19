import numpy as np 
import numpy.typing as npt 
import cv2
from sports.configs import SoccerPitchConfiguration
import supervision as sv
from utils import get_foot_position
import pickle
import os
from ultralytics import YOLO

class ViewTransformer():
    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        model: YOLO,
        frames
    ) -> None:
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(source, target)
        self.CONFIG = SoccerPitchConfiguration()
        self.model = model
        self.frames = frames

    
    def transform_point(self,point):
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.m)
        return tranform_point.reshape(-1,2).astype(np.float32)

    def add_transformed_position_to_tracks(self, tracks):
        keypoints = self.get_all_keypoints(read_from_stub=True, stub_path='stubs/keypoints_stubs.pkl')

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                keypoint = keypoints[frame_num]
                self.mask = (keypoint.xy[0][:, 0] > 1) & (keypoint.xy[0][:, 1] > 1)
                self.source=keypoint.xy[0][self.mask].astype(np.float32)
                self.target=np.array(self.CONFIG.vertices)[self.mask].astype(np.float32)
                
                for track_id, track_info in track.items():
                    position = track_info['position']
                    position = np.array(track_info['position'], dtype=np.float32)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
        with open('stubs/track_stubs.pkl','wb') as f:
            pickle.dump(tracks,f)

    def get_all_keypoints(self, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            
        all_keypoints = []
        for frame_num, frame in enumerate(self.frames):
            result = self.model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            # Store or process the keypoints for this frame
            all_keypoints.append(keypoints)
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(all_keypoints,f)
        
        return all_keypoints

