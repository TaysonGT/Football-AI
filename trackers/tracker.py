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
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            frame_rate=30
        )
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.55,
            minimum_matching_threshold=0.9, 
            lost_track_buffer=30,
            frame_rate=30
        )

        self.last_ball_position = None
        self.ball_missing_frames = 0
        self.MAX_DISTANCE = 200  # Prevent large jumps
        self.MAX_MISSING_FRAMES = 5

    def detect_frames(self, frames, model):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
                # tracks = pickle.load(f)

        detections = self.detect_frames(frames, self.model)
    
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_sv = sv.Detections.from_ultralytics(detection)
            

            # Inside get_object_tracks(), after converting to sv.Detections:
            for object_ind, (class_id, confidence) in enumerate(zip(detection_sv.class_id, detection_sv.confidence)):
                if cls_names[class_id] == "referee" and confidence < 0.7:
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]  # Reclassify as player
                elif cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]  # Existing goalkeeper handling

            # Track Objects
            players_tracked = self.player_tracker.update_with_detections(detection_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            
            for frame_detection in players_tracked:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
        
        for frame_num, detection in enumerate(detections):
            
            detection_sv = sv.Detections.from_ultralytics(detection)

            # Track Objects
            ball_tracked = self.ball_tracker.update_with_detections(detection_sv)

            tracks["ball"].append({})
            
            for frame_detection in ball_tracked:
                bbox = frame_detection[0].tolist()
                tracks["ball"][frame_num][1] = {"bbox":bbox}
        
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

        # rectangle_width = 40
        # rectangle_height=20
        # x1_rect = x_center - rectangle_width//2
        # x2_rect = x_center + rectangle_width//2
        # y1_rect = (y2- rectangle_height//2) +15
        # y2_rect = (y2+ rectangle_height//2) +15

        # if track_id is not None:
        #     cv2.rectangle(frame,
        #                   (int(x1_rect),int(y1_rect) ),
        #                   (int(x2_rect),int(y2_rect)),
        #                   color,
        #                   cv2.FILLED)
            
        #     x1_text = x1_rect+12
        #     if track_id > 99:
        #         x1_text -=10
            
        #     cv2.putText(
        #         frame,
        #         f"{track_id}",
        #         (int(x1_text),int(y1_rect+15)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (0,0,0),
        #         2
        #     )

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

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team1_color, team2_color):
        h, w, _ = frame.shape

        # === 1. Calculate Possession ===
        control_up_to_now = team_ball_control[:frame_num + 1]
        t1_frames = (control_up_to_now == 1).sum()
        t2_frames = (control_up_to_now == 2).sum()
        total = max(t1_frames + t2_frames, 1)

        t1_percent = t1_frames / total

        # === 2. Style Parameters ===
        bar_width = int(w * 0.5)
        bar_height = 30
        x_start = w // 2 - bar_width // 2
        y_start = int(h * 0.92)

        bg_color = (30, 30, 30)
        border_color = (255, 255, 255)

        # === 3. Background bar ===
        cv2.rectangle(frame, (x_start - 2, y_start - 2), (x_start + bar_width + 2, y_start + bar_height + 2), border_color, 1)
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), bg_color, -1)

        # === 4. Fill with team proportions ===
        t1_width = int(bar_width * t1_percent)
        t2_width = bar_width - t1_width

        cv2.rectangle(frame, (x_start, y_start), (x_start + t1_width, y_start + bar_height), team1_color, -1)
        cv2.rectangle(frame, (x_start + t1_width, y_start), (x_start + bar_width, y_start + bar_height), team2_color, -1)

        # === 5. Draw percentages ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        t1_percent = round(t1_percent * 100)
        t2_percent = 100-t1_percent

        # Team 1 %
        percent1_text = f"{int(t1_percent)}%"
        (tw1, th1), _ = cv2.getTextSize(percent1_text, font, font_scale, thickness)
        text_x1 = x_start + t1_width // 2 - tw1 // 2
        text_y = y_start + bar_height // 2 + th1 // 2 - 3
        cv2.putText(frame, percent1_text, (text_x1, text_y), font, font_scale, (255, 255, 255), thickness)

        # Team 2 %
        percent2_text = f"{int(t2_percent)}%"
        (tw2, th2), _ = cv2.getTextSize(percent2_text, font, font_scale, thickness)
        text_x2 = x_start + t1_width + t2_width // 2 - tw2 // 2
        cv2.putText(frame, percent2_text, (text_x2, text_y), font, font_scale, (255, 255, 255), thickness)

        return frame




    def draw_annotations(self,video_frames, tracks,team_ball_control, team1_color, team2_color):
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
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team1_color, team2_color)

            output_video_frames.append(frame)

        return output_video_frames