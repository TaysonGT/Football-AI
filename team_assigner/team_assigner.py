from sklearn.cluster import KMeans
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]

        # Safety check if bbox is too small
        if player_img.size == 0 or player_img.shape[0] < 5 or player_img.shape[1] < 5:
            return np.array([0, 0, 0])

        # Take only the top half (jersey area)
        top_half = player_img[0:player_img.shape[0] // 2, :]

        # Resize to a small size for faster clustering
        top_half = cv2.resize(top_half, (20, 20), interpolation=cv2.INTER_AREA)

        # Convert to LAB color space (better separation of light and color)
        top_half_lab = cv2.cvtColor(top_half, cv2.COLOR_BGR2LAB)
        data = top_half_lab.reshape((-1, 3))

        # Apply KMeans clustering with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
        kmeans.fit(data)
        labels = kmeans.labels_

        clustered_image = labels.reshape(20, 20)

        # Get cluster labels at image corners and sides
        corners = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
            clustered_image[0, 10],  # middle-top
            clustered_image[10, 0],  # middle-left
            clustered_image[10, -1], # middle-right
        ]
        background_cluster = max(set(corners), key=corners.count)

        # Count cluster occurrences
        cluster_counts = np.bincount(labels)

        # Set background cluster to zero so it won't be selected
        cluster_counts[background_cluster] = 0

        # The most common non-background cluster is the jersey
        player_cluster = np.argmax(cluster_counts)

        # Get color of that cluster
        player_color_lab = kmeans.cluster_centers_[player_cluster]

        # Convert LAB back to BGR
        player_color_lab = np.uint8([[player_color_lab]])
        player_color_bgr = cv2.cvtColor(player_color_lab, cv2.COLOR_LAB2BGR)[0][0]

        return player_color_bgr


    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id