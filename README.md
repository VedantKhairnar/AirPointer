# AirPointer Application

## Overview
AirPointer is a touchless mouse application that uses computer vision models to detect hand gestures and keypoints. The application features a graphical user interface (GUI) for real-time webcam-based inference.

## Features
- **Dynamic Model Selection**: Choose between different models for inference.
- **Real-Time Metrics**: Display AI time and FPS during inference.
- **Logging System**: Track application events and errors.
- **User-Friendly UI**: Start/stop webcam and view processed frames with annotations.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AirPointer
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Application
1. Start the GUI:
   ```bash
   python ui.py
   ```
2. Use the dropdown menu to select a model.
3. Click "Start Webcam" to begin inference.
4. View real-time metrics and processed frames.
5. Click "Stop Webcam" to end the session.

### Logs
Application logs are saved in `application.log`.

## Testing

### Unit Tests
Run the test suite:
```bash
pytest tests/
```

## Troubleshooting
- **Webcam not detected**: Ensure your webcam is connected and accessible.
- **Missing weights**: Verify that the model weights are downloaded correctly.
- **Performance issues**: Check system resources and reduce webcam resolution if needed.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License.
