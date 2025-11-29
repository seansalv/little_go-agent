from __future__ import annotations

from fastapi.testclient import TestClient

from go_agent_api.app.main import app

client = TestClient(app)


def make_payload():
	prev = [[0 for _ in range(5)] for _ in range(5)]
	cur = [row[:] for row in prev]
	cur[2][2] = 1
	return {"player": 2, "previousBoard": prev, "currentBoard": cur}


def test_move_endpoint_returns_move(monkeypatch):
	def fake_pick(prev, cur, player, time_limit=None):
		assert player == 2
		return (3, 4)

	monkeypatch.setattr("go_agent_api.app.main.pick_move", fake_pick)
	resp = client.post("/move", json=make_payload())
	assert resp.status_code == 200
	assert resp.json() == {"move": {"row": 3, "col": 4}}


def test_move_endpoint_requires_previous_board_after_moves():
	payload = make_payload()
	payload["previousBoard"] = None
	resp = client.post("/move", json=payload)
	assert resp.status_code == 422


def test_move_endpoint_rejects_bad_board_shape():
	payload = make_payload()
	payload["currentBoard"] = payload["currentBoard"][:4]
	resp = client.post("/move", json=payload)
	assert resp.status_code == 422

