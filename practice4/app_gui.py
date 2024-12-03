from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QWidget, QFormLayout, QComboBox, QTextBrowser, QMenuBar, QAction
)
import sys
import json
import requests


class APIClientGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastAPI Client GUI")

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        self.result_display = QTextBrowser()

        # Menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Categories
        self.categories = {
            "Color Space Conversion": ["/rgb_to_yuv/", "/yuv_to_rgb/"],
            "Image Processing": ["/resize/", "/black_and_white/"],
            "Video Operations": [
                "/change_resolution/",
                "/change_chroma_subsampling/",
                "/get_video_info/",
                "/create_bbb_container/",
                "/video_tracks/",
                "/visualize_motion_vectors/",
                "/generate_yuv_histogram/",
            ],
            "Compression and Encoding": [
                "/dct/encode/",
                "/dct/decode/",
                "/rle/encode/",
                "/rle/decode/",
                "/serpentine/",
                "/convert-video/",
                "/encoding-ladder/",
            ],
        }

        # Endpoint selector
        self.endpoint_selector = QComboBox()
        self.populate_endpoints()
        self.endpoint_selector.currentIndexChanged.connect(self.update_form_fields)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.call_api)

        # Add widgets to layout
        self.main_layout.addWidget(QLabel("Select API Endpoint:"))
        self.main_layout.addWidget(self.endpoint_selector)
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.submit_button)
        self.main_layout.addWidget(QLabel("Result:"))
        self.main_layout.addWidget(self.result_display)

        self.central_widget.setLayout(self.main_layout)

        # Initialize form
        self.update_form_fields()

    def populate_endpoints(self):
        """Populate the dropdown with all API endpoints."""
        for category, endpoints in self.categories.items():
            for endpoint in endpoints:
                self.endpoint_selector.addItem(endpoint)

    def update_form_fields(self):
        """Update the input fields dynamically based on the selected endpoint."""
        # Clear previous form
        for i in reversed(range(self.form_layout.count())):
            self.form_layout.itemAt(i).widget().deleteLater()

        # Get selected endpoint
        endpoint = self.endpoint_selector.currentText()

        # Define fields based on endpoint
        fields, placeholders = self.get_fields_for_endpoint(endpoint)

        # Add input fields with placeholders
        self.input_fields = {}
        for field_name, placeholder in zip(fields, placeholders):
            if field_name in ["resolutions", "codecs", "input_data"]:  # Special handling for JSON or lists
                input_field = QTextEdit()
                input_field.setPlaceholderText(placeholder)
            else:
                input_field = QLineEdit()
                input_field.setPlaceholderText(placeholder)
            self.input_fields[field_name] = input_field
            self.form_layout.addRow(QLabel(field_name), input_field)

    def get_fields_for_endpoint(self, endpoint):
        """Define input fields and placeholders for each endpoint."""
        fields_map = {
            "/rgb_to_yuv/": ["R", "G", "B"],
            "/yuv_to_rgb/": ["Y", "U", "V"],
            "/resize/": ["input_image", "output_image", "width", "height"],
            "/black_and_white/": ["input_image", "output_image"],
            "/change_resolution/": ["input_video", "output_video", "width", "height"],
            "/change_chroma_subsampling/": ["input_video", "output_video", "pixel_format"],
            "/get_video_info/": ["input_video"],
            "/create_bbb_container/": ["input_video", "output_path"],
            "/video_tracks/": ["input_video"],
            "/visualize_motion_vectors/": ["input_video", "output_video"],
            "/generate_yuv_histogram/": ["input_video", "output_video"],
            "/dct/encode/": ["type_of_dct", "input_data"],
            "/dct/decode/": ["type_of_dct", "input_data"],
            "/rle/encode/": ["input_data"],
            "/rle/decode/": ["input_data"],
            "/serpentine/": ["input_data"],
            "/convert-video/": ["input_file", "output_file", "codec"],
            "/encoding-ladder/": ["input_video", "output_video", "codecs", "resolutions"],
        }

        placeholders_map = {
            "/rgb_to_yuv/": ["0-255 for R", "0-255 for G", "0-255 for B"],
            "/yuv_to_rgb/": ["0-255 for Y", "-128 to 127 for U", "-128 to 127 for V"],
            "/resize/": ["/path/to/input.jpg", "/path/to/output.jpg", "Width in pixels", "Height in pixels"],
            "/black_and_white/": ["/path/to/input.jpg", "/path/to/output.jpg"],
            "/change_resolution/": ["/path/to/input.mp4", "/path/to/output.mp4", "New width", "New height"],
            "/change_chroma_subsampling/": ["/path/to/input.mp4", "/path/to/output.mp4", "Pixel format (e.g., YUV420)"],
            "/get_video_info/": ["/path/to/input.mp4"],
            "/create_bbb_container/": ["/path/to/input.mp4", "/path/to/output"],
            "/video_tracks/": ["/path/to/input.mp4"],
            "/visualize_motion_vectors/": ["/path/to/input.mp4", "/path/to/output.mp4"],
            "/generate_yuv_histogram/": ["/path/to/input.mp4", "/path/to/output.mp4"],
            "/dct/encode/": ["DCT type (e.g., 2)", "Comma-separated data (e.g., 1,2,3)"],
            "/dct/decode/": ["DCT type (e.g., 2)", "Comma-separated data (e.g., 1,2,3)"],
            "/rle/encode/": ["Comma-separated integers (e.g., 1,1,2,3,3,3)"],
            "/rle/decode/": ["Comma-separated integers (e.g., 1,2,3)"],
            "/serpentine/": ["Nested array as JSON (e.g., [[1,2],[3,4]])"],
            "/convert-video/": ["/path/to/input.mp4", "/path/to/output.mp4", "Codec (e.g., h264)"],
            "/encoding-ladder/": [
                "/path/to/input.mp4",
                "/path/to/output_base",
                "Comma-separated codecs (e.g., h264, vp9)",
                'JSON array of resolutions (e.g., [{"width":1280,"height":720}])'
            ],
        }

        return fields_map.get(endpoint, []), placeholders_map.get(endpoint, [])

    def call_api(self):
        """Call the selected API endpoint."""
        endpoint = self.endpoint_selector.currentText()
        data = {}

        for field_name, field in self.input_fields.items():
            if isinstance(field, QTextEdit):
                # Parse JSON or list-like inputs
                try:
                    data[field_name] = json.loads(field.toPlainText())
                except json.JSONDecodeError:
                    data[field_name] = field.toPlainText().strip()
            else:
                data[field_name] = field.text().strip()

        # Validate inputs
        if not all(data.values()):
            self.result_display.setText("Error: All fields must be filled.")
            return

        try:
            response = requests.post(f"http://localhost:8000{endpoint}", json=data)
            if response.status_code == 200:
                self.result_display.setText(f"Success:\n{json.dumps(response.json(), indent=4)}")
            else:
                self.result_display.setText(f"Error {response.status_code}:\n{response.text}")
        except Exception as e:
            self.result_display.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = APIClientGUI()
    window.show()
    sys.exit(app.exec_())

