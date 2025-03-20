import cv2
import matplotlib.pyplot as plt

class DetectionVisualizer:
    @staticmethod
    def visualize_detections(thermal_img, visual_img, detections):
        """Visualize detections from all modalities"""
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[4, 1, 1])
        
        # Thermal image subplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Draw detections on thermal image
        thermal_rgb = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
        for (x1, y1, x2, y2) in detections['thermal_contours']:
            cv2.rectangle(thermal_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(thermal_rgb, 'Human', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ax1.imshow(thermal_rgb)
        ax1.set_title(f'Thermal Detections: {detections["thermal_detections"]}')
        
        # Visual image subplot
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Draw detections on visual image
        visual_rgb = cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB)
        for (x1, y1, x2, y2) in detections['visual_detections']:
            cv2.rectangle(visual_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visual_rgb, 'Person', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ax2.imshow(visual_rgb)
        ax2.set_title(f'Visual Detections: {len(detections["visual_detections"])}')
        
        # Audio detection subplot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        if detections['audio_alert']['distress_detected']:
            ax3.text(0.5, 0.5, '🔊 AUDIO ALERT: Distress Sound Detected!', 
                    color='red', fontsize=14, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='yellow', alpha=0.3, edgecolor='red'))
        else:
            ax3.text(0.5, 0.5, '🔊 No Audio Distress Detected', 
                    color='green', fontsize=14,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='green'))
        
        # Emotion detection subplot
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        emotions = detections['audio_alert']['emotions']
        emotion_text = "Detected Emotions: "
        emotion_colors = []
        emotion_scores = []
        
        for emotion, score in emotions.items():
            if score > 0.1:  # Only show significant emotions
                emotion_text += f"{emotion.capitalize()} ({score:.0%}), "
                emotion_colors.append('red' if score > 0.5 else 'orange')
                emotion_scores.append(score)
        
        if emotion_scores:
            ax4.text(0.5, 0.5, emotion_text[:-2],  # Remove last comma
                    color='black', fontsize=12, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'))
        
        plt.tight_layout()
        return fig 