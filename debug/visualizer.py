import cv2


class DebugVisualizer:
    def draw_state_info(self, frame, state_info, temporal_features):
        cv2.rectangle(frame, (10, 10), (460, 210), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (460, 210), (255, 255, 255), 1)

        lines = [
            f"State: {state_info.current_state.name}",
            f"Prev: {state_info.previous_state.name}",
            f"Duration: {state_info.state_duration:.2f}s",
            f"Action: {state_info.pending_action}",
            f"Confidence: {temporal_features.confidence:.2f}",
            f"Vel: {temporal_features.palm_velocity_trend:.4f}",
            f"Stability: {temporal_features.palm_stability_trend:.2f}",
            f"Pinched: {temporal_features.is_pinched}",
            f"Moving: {temporal_features.is_moving}",
        ]

        y = 35
        for line in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            y += 20

        return frame

    def draw_palm_indicator(self, frame, hand_features):
        h, w = frame.shape[:2]
        x = int(hand_features.palm_x * w)
        y = int(hand_features.palm_y * h)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)

        dx = int(hand_features.palm_velocity * 25)
        cv2.arrowedLine(frame, (x, y), (x + dx, y), (255, 0, 0), 2, tipLength=0.25)
        return frame

    def draw_finger_states(self, frame, hand_features):
        states = [
            ("Thumb", hand_features.thumb_extended),
            ("Index", hand_features.index_extended),
            ("Middle", hand_features.middle_extended),
            ("Ring", hand_features.ring_extended),
            ("Pinky", hand_features.pinky_extended),
        ]

        x = 10
        y = frame.shape[0] - 120
        cv2.rectangle(frame, (x, y - 25), (290, y + 90), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y - 25), (290, y + 90), (255, 255, 255), 1)

        for idx, (name, active) in enumerate(states):
            color = (0, 255, 0) if active else (0, 0, 255)
            label = f"{name}: {'EXT' if active else 'FLEX'}"
            cv2.putText(
                frame,
                label,
                (x + 10, y + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return frame

    def draw_all_debug_info(self, frame, hand_features, temporal_features, state_info):
        frame = self.draw_state_info(frame, state_info, temporal_features)
        frame = self.draw_palm_indicator(frame, hand_features)
        frame = self.draw_finger_states(frame, hand_features)
        return frame
