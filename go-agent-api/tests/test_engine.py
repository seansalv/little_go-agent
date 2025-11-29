from go_agent_api.go_engine import BLACK, WHITE, pick_move


def empty_board():
	return [[0 for _ in range(5)] for _ in range(5)]


def full_board():
	board = []
	for i in range(5):
		row = []
		for j in range(5):
			row.append(1 if (i + j) % 2 == 0 else 2)
		board.append(row)
	return board


def test_pick_move_returns_valid_coordinates_on_empty_board():
	move = pick_move(None, empty_board(), BLACK, time_limit=0.2)
	assert move is not None
	row, col = move
	assert 0 <= row < 5
	assert 0 <= col < 5


def test_pick_move_passes_when_no_intersections_open():
	board = full_board()
	move = pick_move(board, board, WHITE, time_limit=0.2)
	assert move is None

