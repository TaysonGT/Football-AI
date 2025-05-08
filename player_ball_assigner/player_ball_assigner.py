import numpy as np
from collections import defaultdict
import math

class PlayerBallAssigner:
    def __init__(self, max_distance=200, min_control_frames=5, movement_threshold=100):
        """
        Args:
            max_distance: Maximum pixel distance between player feet and ball (default: 70)
            min_control_frames: Consecutive frames needed to confirm possession (default: 3)
            movement_threshold: Pixel movement to reset possession (default: 20)
        """
        self.max_player_ball_distance = max_distance
        self.min_control_frames = min_control_frames
        self.movement_threshold = movement_threshold
        self.player_history = defaultdict(lambda: {
            'contact_frames': 0,
            'last_position': None,
            'last_frame': -1
        })

    def assign_ball_to_player(self, players, ball_pos, frame_num):
        """
        Args:
            players: Dict {player_id: {'bbox': [x1,y1,x2,y2]}}
            ball_pos: (x,y) or None
            frame_num: Current frame number
            
        Returns:
            player_id or None if no valid player
        """
        if ball_pos is None:
            return None

        candidates = []

        for player_id, player in players.items():
            feet_pos = player['position_transformed']
            distance = self._euclidean_distance(feet_pos, ball_pos)

            if distance <= self.max_player_ball_distance:
                candidates.append({
                    'player_id': player_id,
                    'distance': distance,
                    'feet_pos': feet_pos
                })

        if not candidates:
            return None

        # Select closest candidate
        best_candidate = min(candidates, key=lambda x: x['distance'])
        player_id = best_candidate['player_id']
        
        # Update history and check possession
        self._update_player_history(player_id, best_candidate['feet_pos'], frame_num)
        
        if self.player_history[player_id]['contact_frames'] >= self.min_control_frames:
            return player_id
        return None

    def reset(self):
        """Clear tracking history"""
        self.player_history.clear()

    def _euclidean_distance(self, pos1, pos2):
        """Calculate distance between two (x,y) points"""
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def _update_player_history(self, player_id, feet_pos, frame_num):
        """Update player tracking history with movement validation"""
        history = self.player_history[player_id]
        
        # Reset if player moved significantly or frame isn't consecutive
        if (history['last_position'] and 
            (self._euclidean_distance(history['last_position'], feet_pos) > self.movement_threshold or
             frame_num != history['last_frame'] + 1)):
            history['contact_frames'] = 0
        
        # Update position
        history['last_position'] = feet_pos
        history['last_frame'] = frame_num
        history['contact_frames'] += 1