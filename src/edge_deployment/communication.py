"""Communication layer: REST server and MQTT client for edge deployment."""

from __future__ import annotations

import json
import warnings

import numpy as np


class RESTServer:
    """Lightweight REST server for trajectory inference.

    Parameters
    ----------
    inference_session : ONNXInferenceSession or similar
    host : str
    port : int
    """

    def __init__(self, inference_session, host: str = "127.0.0.1", port: int = 8080):
        self.session = inference_session
        self.host = host
        self.port = port
        self._app = None

    def _create_app(self):
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError("Flask is required for REST server. Install via: pip install flask")

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        def predict():
            data = request.get_json()
            condition = np.array(data["condition"], dtype=np.float32)
            if condition.ndim == 2:
                condition = condition[np.newaxis, :]
            trajectory = self.session.predict(condition)
            return jsonify({"trajectory": trajectory.tolist()})

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"})

        self._app = app
        return app

    def start(self):
        """Start the REST server."""
        app = self._create_app()
        app.run(host=self.host, port=self.port)


class MQTTClient:
    """MQTT client for IoT trajectory inference.

    Parameters
    ----------
    broker : str
    topic : str
    inference_session : ONNXInferenceSession or similar
    """

    def __init__(self, broker: str, topic: str, inference_session):
        self.broker = broker
        self.topic = topic
        self.session = inference_session
        self._client = None

    def start(self):
        """Start MQTT client and subscribe to condition topic."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("paho-mqtt is required for MQTT. Install via: pip install paho-mqtt")

        client = mqtt.Client()

        def on_message(client, userdata, msg):
            data = json.loads(msg.payload.decode())
            condition = np.array(data["condition"], dtype=np.float32)
            if condition.ndim == 2:
                condition = condition[np.newaxis, :]
            trajectory = self.session.predict(condition)
            response_topic = f"{self.topic}/response"
            client.publish(response_topic, json.dumps({"trajectory": trajectory.tolist()}))

        client.on_message = on_message
        client.connect(self.broker)
        client.subscribe(self.topic)
        self._client = client
        client.loop_start()

    def stop(self):
        """Stop the MQTT client."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
