from utils import read_video, save_video
from trackers import Tracker
from ultralytics import YOLO
import supervision as sv
import os
import sys
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()


def process_video(video_path, send_progress_callback):
    # Read Video
    video_frames = read_video(video_path)
    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)  # Removes extension
    output_video = f"output_videos/processed_{name}.mp4"

    # Initialize Tracker
    send_progress_callback("Initializing Tracker...", 20)
    tracker = Tracker('models/3k_imgs.pt')
    keypoints_model = YOLO('models/pitch_keypoints.pt')

    send_progress_callback("Tracking objects...", 30)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # View Transformer
    send_progress_callback("Transforming View...", 40)
    result = keypoints_model(video_frames[0], verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(result)
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    view_transformer = ViewTransformer(
        source= keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        model= keypoints_model,
        frames= video_frames
    )
    view_transformer.add_transformed_position_to_tracks(tracks)


    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    send_progress_callback("Estimating Speed and Distance...", 50)
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    send_progress_callback("Assigning Teams...", 60)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    send_progress_callback("Possession Estimator...", 70)
    
    player_assigner = PlayerBallAssigner()

    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        if 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
        else:
            ball_bbox = None  # Or use a predicted value

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox, frame_num)

        if assigned_player != -1 and assigned_player != None:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if(len(team_ball_control)>0):
                team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    teams, possession = np.unique(team_ball_control, return_counts=True)   
    teams_control = dict(zip(teams, possession))

    t1 = teams_control.get(1,0)
    t2 = teams_control.get(2,0)

    t1 = round((t1/(t1+t2)) * 100)
    t2 = 100 - t1

    # Draw output 
    send_progress_callback("Drawing Annotations...", 80)

    ## Draw object Tracks
    team_colors = team_assigner.team_colors
    team1_color = tuple(map(int, team_colors[1]))
    team2_color = tuple(map(int, team_colors[2]))
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control, team1_color, team2_color)


    ## Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    ## Draw Camera movement
    # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Save video
    send_progress_callback("Exporting Output Video...", 90)
    save_video(output_video_frames, output_video)
    send_final_possession(
        t1, 
        t2, 
        c1=(team1_color[2], team1_color[1], team1_color[0]), 
        c2=(team2_color[2], team2_color[1], team2_color[0])
    )
    send_progress_callback("Processing Complete!", 100)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: No video file provided.")
        sys.exit(1)
    
    def send_progress_callback(msg, pct):
        print(f"PROGRESS:{pct}:{msg}", flush=True)  # Special format
    
    def send_final_possession(t1, t2, c1, c2):
        print(f"FinalPossession:{t1}:{t2}:{c1}:{c2}", flush=True)  # Special format

    video = sys.argv[1]
    process_video(video, send_progress_callback)
