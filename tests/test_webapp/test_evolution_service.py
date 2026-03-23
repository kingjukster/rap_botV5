"""Tests for webapp.services.evolution_service."""

import os

from webapp.services.evolution_service import _can_use_docker


def test_can_use_docker_false_when_no_socket(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda p: False if p == "/var/run/docker.sock" else True)
    monkeypatch.setenv("RAPBOT_DOCKER_NETWORK", "rap_botv5_default")
    assert _can_use_docker() is False


def test_can_use_docker_false_when_no_network_env(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda p: True if p == "/var/run/docker.sock" else False)
    monkeypatch.delenv("RAPBOT_DOCKER_NETWORK", raising=False)
    assert _can_use_docker() is False


def test_can_use_docker_true_when_socket_and_network(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/var/run/docker.sock")
    monkeypatch.setenv("RAPBOT_DOCKER_NETWORK", "rap_botv5_default")
    assert _can_use_docker() is True
