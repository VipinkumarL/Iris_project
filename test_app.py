import os
import tempfile
import unittest
from unittest.mock import patch

import app


def sample_pattern():
    sector_features = []
    for map_index in range(3):
        for band_index in range(4):
            for sector_index in range(12):
                sector_features.append(120 + (map_index * 140) + (band_index * 31) + (sector_index * 13))
    transition_features = [180 + ((index * 11) % 220) for index in range(56)]
    continuous_features = sector_features + transition_features
    iris_bits = [1000 if (index % 5 in (0, 2, 3)) else 0 for index in range(192)]
    raw = [500, 520, 150] + continuous_features + iris_bits
    return build_pattern(raw)


def roll_raw_iris_sections(raw_values, sector_shift):
    geometry = raw_values[:3]
    texture = raw_values[3:-app.IRIS_CODE_LENGTH]
    bits = raw_values[-app.IRIS_CODE_LENGTH:]

    texture_array = app.np.asarray(texture, dtype=app.np.float32)
    bits_array = app.np.asarray(bits, dtype=app.np.float32)
    shifted_texture = app._roll_sector_texture(texture_array, sector_shift).astype(int).tolist()
    shifted_bits = app._roll_iris_bits(bits_array, sector_shift).astype(int).tolist()
    return geometry + shifted_texture + shifted_bits


def build_pattern(raw_values):
    deltas = [abs(raw_values[index + 1] - raw_values[index]) for index in range(len(raw_values) - 1)]
    return raw_values + deltas


class IrisAppTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        app.DATABASE_PATH = self.db_path
        app.init_db(self.db_path)
        app.app.config["TESTING"] = True
        app.app.config["PROPAGATE_EXCEPTIONS"] = False
        self.client = app.app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_compare_patterns_returns_full_score_for_identical_patterns(self):
        pattern = sample_pattern()
        score = app.compare_patterns(pattern, pattern)
        self.assertGreaterEqual(score, 0.99)

    def test_compare_patterns_rejects_large_geometry_shift(self):
        reference = sample_pattern()
        candidate = sample_pattern()
        candidate[0] = 900
        candidate[1] = 920
        candidate[2] = 890

        self.assertEqual(app.compare_patterns(reference, candidate), 0.0)

    def test_compare_patterns_accepts_moderate_webcam_position_shift(self):
        reference_raw = sample_pattern()[:395]
        candidate_raw = reference_raw[:]
        candidate_raw[0] = 555
        candidate_raw[1] = 585
        candidate_raw[2] = 175

        self.assertGreaterEqual(
            app.compare_patterns(build_pattern(reference_raw), build_pattern(candidate_raw)),
            app.MATCH_THRESHOLD,
        )

    def test_compare_patterns_accepts_small_texture_noise_from_same_capture(self):
        reference_raw = sample_pattern()[:396]
        candidate_raw = reference_raw[:]
        for index in range(3, 203):
            candidate_raw[index] += 8 if index % 2 == 0 else -8
        for index in range(203, 223):
            candidate_raw[index] = 1000 - candidate_raw[index]

        score = app.compare_patterns(build_pattern(reference_raw), build_pattern(candidate_raw))

        self.assertGreaterEqual(score, app.MATCH_THRESHOLD)

    def test_compare_patterns_rejects_different_iris_code(self):
        reference_raw = sample_pattern()[:396]
        candidate_raw = reference_raw[:]
        for index in range(3, 203):
            candidate_raw[index] = 700 - (candidate_raw[index] // 2)
        for index in range(203, 395):
            candidate_raw[index] = 1000 - candidate_raw[index]

        self.assertEqual(app.compare_patterns(build_pattern(reference_raw), build_pattern(candidate_raw)), 0.0)

    def test_compare_biometrics_accepts_matching_visual_signature(self):
        pattern = sample_pattern()
        signature = [120 + (index % 40) for index in range(768)]

        score = app.compare_biometrics(pattern, pattern, signature, signature)

        self.assertGreaterEqual(score, app.MATCH_THRESHOLD)

    def test_compare_biometrics_accepts_strong_pattern_when_visual_signature_shifts(self):
        pattern = sample_pattern()
        reference_signature = [120 + (index % 40) for index in range(768)]
        candidate_signature = [220 - (index % 35) for index in range(768)]

        score = app.compare_biometrics(pattern, pattern, reference_signature, candidate_signature)

        self.assertGreaterEqual(score, app.MATCH_THRESHOLD)

    def test_compare_biometrics_does_not_veto_medium_pattern_with_weak_signature(self):
        reference_raw = sample_pattern()[:396]
        candidate_raw = reference_raw[:]
        for index in range(3, 50):
            candidate_raw[index] += 20 if index % 2 == 0 else -20
        for index in range(203, 278):
            candidate_raw[index] = 1000 - candidate_raw[index]

        reference_pattern = build_pattern(reference_raw)
        candidate_pattern = build_pattern(candidate_raw)
        pattern_score = app.compare_patterns(reference_pattern, candidate_pattern)
        self.assertGreaterEqual(pattern_score, app.MATCH_THRESHOLD)

        reference_signature = [120 + (index % 40) for index in range(768)]
        candidate_signature = [220 - (index % 35) for index in range(768)]
        signature_score = app.compare_visual_signatures(reference_signature, candidate_signature)
        self.assertLess(signature_score, 0.65)

        score = app.compare_biometrics(
            reference_pattern,
            candidate_pattern,
            reference_signature,
            candidate_signature,
        )

        self.assertGreaterEqual(score, app.MATCH_THRESHOLD)

    def test_build_iris_password_returns_error_when_processing_fails(self):
        handle, image_path = tempfile.mkstemp(suffix=".jpg")
        os.close(handle)

        with patch("app.save_image_from_data_url", return_value=image_path):
            with patch("app.assess_image_quality", side_effect=RuntimeError("opencv failed")):
                iris_data, error = app.build_iris_password("data:image/jpeg;base64,aaa")

        self.assertIsNone(iris_data)
        self.assertIn("could not process", error)
        self.assertFalse(os.path.exists(image_path))

    def test_build_iris_password_rejects_non_string_capture_payload(self):
        iris_data, error = app.build_iris_password({"image": "not-a-data-url"})

        self.assertIsNone(iris_data)
        self.assertIn("Camera image was not received", error)

    def test_compare_biometrics_rejects_strong_visual_match_when_pattern_is_unstable(self):
        reference = sample_pattern()
        candidate = sample_pattern()
        candidate[0] = 900
        candidate[1] = 920
        candidate[2] = 890
        signature = [120 + (index % 40) for index in range(768)]

        self.assertEqual(app.compare_biometrics(reference, candidate, signature, signature), 0.0)

    def test_compare_biometrics_rejects_weak_visual_match(self):
        reference = sample_pattern()
        candidate = sample_pattern()
        candidate[0] = 900
        candidate[1] = 920
        candidate[2] = 890
        reference_signature = [120 + (index % 40) for index in range(768)]
        candidate_signature = [220 - (index % 35) for index in range(768)]

        self.assertEqual(
            app.compare_biometrics(reference, candidate, reference_signature, candidate_signature),
            0.0,
        )

    def test_compare_patterns_accepts_small_iris_rotation(self):
        reference_raw = sample_pattern()[:396]
        candidate_raw = roll_raw_iris_sections(reference_raw, 2)

        score = app.compare_patterns(build_pattern(reference_raw), build_pattern(candidate_raw))

        self.assertGreaterEqual(score, app.MATCH_THRESHOLD)

    def test_api_health_exposes_service_metadata(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["app_name"], app.APP_NAME)
        self.assertEqual(payload["service"], app.API_SERVICE_NAME)

    def test_api_preflight_requests_return_empty_no_content_response(self):
        for endpoint in ("/api/register", "/api/login", "/api/session/verify"):
            with self.subTest(endpoint=endpoint):
                response = self.client.options(endpoint)

                self.assertEqual(response.status_code, 204)
                self.assertEqual(response.get_data(as_text=True), "")

    def test_api_register_persists_user_when_biometrics_are_valid(self):
        iris_payload = {
            "hash": "abc123",
            "pattern": sample_pattern(),
            "quality": {"ok": True},
            "circles": [(10, 10, 5)],
            "features": [(0.1, 0.2, 0.3)],
        }

        with patch("app.build_iris_password", return_value=(iris_payload, None)):
            response = self.client.post(
                "/api/register",
                json={"username": "vipin", "image_data": "data:image/jpeg;base64,aaa"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIsNotNone(app.fetch_user_record("vipin"))

    def test_api_register_rejects_non_object_json_payload(self):
        response = self.client.post("/api/register", json=["username", "vipin"])

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["message"], "Username is required")

    def test_api_login_returns_token_for_matching_pattern(self):
        pattern = sample_pattern()
        app.register_user("vipin", "abc123", pattern)
        iris_payload = {
            "hash": "def456",
            "pattern": pattern,
            "quality": {"ok": True},
            "circles": [(10, 10, 5)],
            "features": [(0.1, 0.2, 0.3)],
        }

        with patch("app.build_iris_password", return_value=(iris_payload, None)):
            response = self.client.post(
                "/api/login",
                json={"username": "vipin", "image_data": "data:image/jpeg;base64,aaa"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("auth_token", payload)

    def test_database_is_recreated_after_local_file_is_deleted(self):
        os.remove(self.db_path)

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(os.path.exists(self.db_path))
        self.assertIsNone(app.fetch_user_record("missing-user"))

    def test_api_session_verify_accepts_signed_token(self):
        token = app.issue_auth_token("vipin")
        response = self.client.post("/api/session/verify", json={"auth_token": token})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["username"], "vipin")

    def test_page_errors_show_friendly_message_instead_of_internal_server_error(self):
        with patch("app.fetch_user_record", side_effect=RuntimeError("database failed")):
            response = self.client.post(
                "/login",
                data={"username": "vipin", "image_data": "data:image/jpeg;base64,aaa"},
            )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 500)
        self.assertIn("Something went wrong. Please try again.", body)
        self.assertNotIn("Internal Server Error", body)

    def test_api_errors_return_json_instead_of_internal_server_error(self):
        with patch("app.fetch_user_record", side_effect=RuntimeError("database failed")):
            response = self.client.post(
                "/api/login",
                json={"username": "vipin", "image_data": "data:image/jpeg;base64,aaa"},
            )

        body = response.get_data(as_text=True)
        payload = response.get_json()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["message"], "Something went wrong. Please try again.")
        self.assertNotIn("Internal Server Error", body)

    def test_missing_page_uses_friendly_error_page(self):
        response = self.client.get("/missing-page")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 404)
        self.assertIn("Request could not be completed", body)
        self.assertNotIn("Internal Server Error", body)


if __name__ == "__main__":
    unittest.main()
