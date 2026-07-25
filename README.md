
# IrisPass

IrisPass is a Flask-based biometric authentication demo that uses a webcam capture, OpenCV iris localization, handcrafted iris texture features, and a lightweight pattern-matching pipeline to simulate iris login.

The project is meant for academic demonstration and learning. It shows the end-to-end workflow of biometric enrollment and verification, but it is not a production-grade iris security system.

## What The Project Does

1. A user enters a username and captures an eye image in the browser.
2. The backend validates the image quality and checks that the frame contains a usable subject.
3. The iris region is detected from the uploaded eye image.
4. Texture features are extracted from the detected iris circle.
5. The feature vector is transformed into a stored comparison pattern.
6. During login, a newly captured pattern is compared against the enrolled one.
7. If the similarity score crosses the configured threshold, the user is authenticated.

## Tech Stack

- OpenCV
- Flask
-Numpu
- Python
- SQLite
- HTML, CSS, and browser camera APIs

## Project Structure

- `app.py` contains Flask routes, biometric enrollment/login flow, token handling, and API responses.
- `iris_processing.py` contains iris localization, eye-region selection, and image quality checks.
- `feature_extraction.py` converts a detected iris circle into normalized texture features.
- `constellation.py` converts extracted features into a compact comparison pattern.
- `capture.py` decodes browser-captured base64 images into temporary files.
- `database.py` initializes and accesses the SQLite user store.
- `templates/` contains the registration, login, dashboard, and integration demo screens.
- `tests/test_app.py` contains lightweight automated tests for matching logic and API behavior.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Running Tests

Run the built-in unit tests with:

```bash
python -m unittest discover -s tests
```

These tests cover:

- pattern comparison behavior
- API health metadata
- registration flow with mocked biometric extraction
- login token generation
- signed session verification

## API Endpoints

- `GET /api/health`
- `GET /api/meta`
- `POST /api/register`
- `POST /api/login`
- `POST /api/session/verify`

Example register/login payload:

```json
{
  "username": "vipin",
  "image_data": "data:image/jpeg;base64,..."
}
```

Example token verification payload:

```json
{
  "auth_token": "signed-token-from-login"
}
```

## Demo Notes

- Camera capture works best on `localhost` or HTTPS.
- Good lighting and a steady face position improve iris detection.
- Re-register users if you change the feature extraction or matching logic.
- Delete local `.db` files before sharing the project if you do not want to ship test users.

## Presentation Angle

If you are presenting this project, a clean and accurate description is:

`A prototype iris-based authentication system that combines webcam capture, image preprocessing, handcrafted biometric features, and Flask-based verification APIs.`

Good phrases to use:

- prototype biometric authentication system
- academic/demo implementation
- feature-based iris verification workflow
- Flask API with browser-side image capture

Claims to avoid:

- enterprise-grade biometric security
- commercial iris recognition accuracy
- foolproof real-world authentication

## Current Limitations

- The system uses a simplified iris-matching approach instead of a production biometric model.
- Accuracy depends heavily on lighting, framing, and image sharpness.
- SQLite is used for demo simplicity and is not tuned for multi-user deployment.
- The stored pattern format is useful for demonstration but not a secure biometric template standard.
