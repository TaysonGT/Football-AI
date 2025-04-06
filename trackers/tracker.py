from ultralytics import YOLO
import supervision as sv
import pickle
import os
from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, player_model_path, ball_model_path):
        self.player_model = YOLO(player_model_path) 
        self.ball_model = YOLO(ball_model_path) 

        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            frame_rate=30
        )
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.2, 
            minimum_matching_threshold=0.8, 
            lost_track_buffer=30,
            frame_rate=30
        )
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],  # State transition model
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                              [0, 1, 0, 0]])
        self.kf.P *= 1000  # Increase covariance
        self.kf.Q *= 0.01  # Process noise
        self.kf.R *= 10  # Measurement noise

        self.last_ball_position = None
        self.ball_missing_frames = 0
        self.MAX_DISTANCE = 200  # Prevent large jumps
        self.MAX_MISSING_FRAMES = 5

    def detect_frames(self, frames, model):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = model.predict(frames[i:i+batch_size],conf=0.2)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        player_detections = self.detect_frames(frames, self.player_model)
        ball_detections = self.detect_frames(frames, self.ball_model)

    
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, (player_detection, ball_detection) in enumerate(zip(player_detections, ball_detections)):
            cls_names = player_detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            player_detection_sv = sv.Detections.from_ultralytics(player_detection)
            ball_detection_sv = sv.Detections.from_ultralytics(ball_detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(player_detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    player_detection_sv.class_id[object_ind] = cls_names_inv["player"]
                    
            print(f"[Frame {frame_num}] Ball Detections (raw):")
            for i, det in enumerate(ball_detection_sv):
                xyxy, mask, conf, class_id, tracker_id, class_name = det
                print(f"  [{i}] Box: {xyxy.tolist()}, Conf: {conf:.2f}, Class ID: {class_id}, Name: {class_name}")

            # Track Objects
            # Filter detections for the ball
            players_tracked = self.player_tracker.update_with_detections(player_detection_sv)
            ball_tracked = self.ball_tracker.update_with_detections(ball_detection_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in players_tracked:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in ball_tracked:
                bbox = frame_detection[0].tolist()
                tracks["ball"][frame_num][1] = {"bbox":bbox}
            print(f"[Frame {frame_num}] Ball Tracks: {[track[0].tolist() for track in ball_tracked]}")

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        h, w, _ = frame.shape
        
        rect_x1, rect_y1 = int(0.7 * w), int(0.85 * h)  # Bottom-right corner
        rect_x2, rect_y2 = int(0.98 * w), int(0.98 * h)
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2,rect_y2), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
           
            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}


            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames