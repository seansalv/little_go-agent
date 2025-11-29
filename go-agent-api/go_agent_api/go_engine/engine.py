from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import List, Optional, Tuple

N = 5
EMPTY, BLACK, WHITE = 0, 1, 2
KOMI = 2.5
DEFAULT_TIME_LIMIT = 8.6

Board = List[List[int]]
Move = Optional[Tuple[int, int]]


def resolve_time_limit(value: Optional[float]) -> float:
	"""Clamp external time limits to a sane, positive window."""
	if value is None:
		return DEFAULT_TIME_LIMIT
	try:
		limit = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError("time limit must be a float") from exc
	return max(0.2, min(limit, 30.0))


def env_time_limit() -> float:
	"""Read ENGINE_TIME_LIMIT from the environment or fallback."""
	raw = os.getenv("ENGINE_TIME_LIMIT")
	if raw is None:
		return DEFAULT_TIME_LIMIT
	try:
		return resolve_time_limit(raw)
	except ValueError:
		return DEFAULT_TIME_LIMIT

__all__ = [
	"pick_move",
	"Searcher",
	"N",
	"EMPTY",
	"BLACK",
	"WHITE",
]

# I/O stuff
def read_input(path="input.txt"):
	with open(path, "r") as f:
		me = int(f.readline().strip())
		prev = [list(map(int, list(f.readline().strip()))) for _ in range(N)]
		cur = [list(map(int, list(f.readline().strip()))) for _ in range(N)]
	return me, prev, cur

def write_output(move, path="output.txt"):
	with open(path, "w") as f:
		if move is None:
			f.write("PASS\n")
		else:
			i, j = move
			f.write(f"{i},{j}\n")

# board utils
def neighbors(i, j):
	for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
		ni, nj = i+di, j+dj
		if 0 <= ni < N and 0 <= nj < N:
			yield ni, nj

def collect_group(board, i, j):
	color = board[i][j]
	stack = [(i,j)]
	seen = {(i,j)}
	while stack:
		x, y = stack.pop()
		for nx, ny in neighbors(x, y):
			if board[nx][ny] == color and (nx,ny) not in seen:
				seen.add((nx,ny))
				stack.append((nx,ny))
	return seen

def group_liberties(board, group):
	libs = set()
	for x, y in group:
		for nx, ny in neighbors(x, y):
			if board[nx][ny] == EMPTY:
				libs.add((nx,ny))
	return libs

def remove_group(board, group):
	for x, y in group:
		board[x][y] = EMPTY

def simulate_play(board, move, color):
	if move is None:
		return deepcopy(board), 0
	i, j = move
	if board[i][j] != EMPTY:
		return None, 0
	newb = [r[:] for r in board]
	newb[i][j] = color

	opp = 3 - color
	captured = 0
	to_remove = []
	seen = set()

	for ni, nj in neighbors(i, j):
		if newb[ni][nj] == opp and (ni,nj) not in seen:
			g = collect_group(newb, ni, nj)
			seen |= g
			if len(group_liberties(newb, g)) == 0:
				to_remove.append(g)
	for g in to_remove:
		captured += len(g)
		remove_group(newb, g)

	my_group = collect_group(newb, i, j)
	if len(group_liberties(newb, my_group)) == 0 and captured == 0:
		return None, 0
	return newb, captured

def boards_equal(a, b):
	for i in range(N):
		if a[i] != b[i]:
			return False
	return True

def legal_moves(prev_board, cur_board, color):
	moves = []
	for i in range(N):
		for j in range(N):
			if cur_board[i][j] != EMPTY:
				continue
			nb, _ = simulate_play(cur_board, (i,j), color)
			if nb is None:
				continue
			if prev_board is not None and boards_equal(nb, prev_board):
				continue
			moves.append((i,j))
	moves.append(None)
	return moves

# eval
def evaluate(board, color, move_count):
	opp = 3 - color
	my_stones = sum(c == color for r in board for c in r)
	opp_stones = sum(c == opp for r in board for c in r)
	base = my_stones - opp_stones + (KOMI if color == WHITE else 0.0)

	def all_groups(c):
		seen, groups = set(), []
		for i in range(N):
			for j in range(N):
				if board[i][j] == c and (i,j) not in seen:
					g = collect_group(board, i, j)
					seen |= g
					groups.append(g)
		return groups

	my_lib = sum(len(group_liberties(board, g)) for g in all_groups(color))
	opp_lib = sum(len(group_liberties(board, g)) for g in all_groups(opp))

	pos_bonus = 0.0
	fade = max(0.0, 1.0 - move_count / 12.0)
	if fade > 0:
		for i in range(N):
			for j in range(N):
				if board[i][j] == color:
					if (i in (0, N-1)) and (j in (0, N-1)):
						pos_bonus += 0.15 * fade
					elif i in (0, N-1) or j in (0, N-1):
						pos_bonus += 0.05 * fade
	return base + 0.5 * (my_lib - opp_lib) + pos_bonus

def ordering_score(board, move, color):
	if move is None: return -1
	nb, captured = simulate_play(board, move, color)
	if nb is None: return -1e9
	score = 5 * captured
	opp = 3 - color
	for ni, nj in neighbors(*move):
		if nb[ni][nj] == opp:
			g = collect_group(nb, ni, nj)
			if len(group_liberties(nb, g)) == 1:
				score += 1
	g = collect_group(nb, move[0], move[1])
	if len(group_liberties(nb, g)) == 1:
		score -= 1.5
	i, j = move
	if (i in (0, N-1)) and (j in (0, N-1)): score += 0.1
	elif i in (0, N-1) or j in (0, N-1): score += 0.05
	return score

def capture_moves(prev_board, cur_board, color):
	caps = []
	for m in legal_moves(prev_board, cur_board, color):
		if m is None: continue
		nb, captured = simulate_play(cur_board, m, color)
		if nb is None: continue
		if captured > 0:
			caps.append((m, nb))
		else:
			opp = 3 - color
			i, j = m
			for ni, nj in neighbors(i, j):
				if nb[ni][nj] == opp:
					g = collect_group(nb, ni, nj)
					if len(group_liberties(nb, g)) == 1:
						caps.append((m, nb))
						break
	return caps

class Searcher:
	def __init__(self, me: int, prev_board: Optional[Board], cur_board: Board, time_limit: Optional[float] = None):
		self.me = me
		self.prev = prev_board
		self.start_board = cur_board
		self.start_time = time.perf_counter()
		self.time_limit = resolve_time_limit(time_limit) if time_limit is not None else env_time_limit()
		self.best_move = None
		self.move_count = sum(c != EMPTY for r in cur_board for c in r)
		self.TT = {}
		self.FLAG_EXACT, self.FLAG_LOWER, self.FLAG_UPPER = 0, 1, 2
		self.max_depth = 5
		if self.me == BLACK and self.move_count < 2:
			self.max_depth = min(self.max_depth + 1, 7)
			#self.time_limit += 0.8
		self.killers = [[None, None] for _ in range(64)]
		self.history = {}
		self.pv = {}
		self.nodes = 0

	def time_up(self):
		return (time.perf_counter() - self.start_time) >= self.time_limit
	
	def key(self, prev_b, cur_b, to_play):
		return (tuple(map(tuple, prev_b)) if prev_b else None,
				tuple(map(tuple, cur_b)), to_play)

	def move_id(self, move):
		if move is None: return N*N
		return move[0]*N + move[1]

	def search(self):
		depth = 1
		last_score = 0
		last_best = None
		while depth <= self.max_depth:
			if self.time_up(): break
			window = (0.5 + 0.1*depth) if depth > 1 else 1e6
			alpha, beta = last_score - window, last_score + window
			score, bm = self.alpha_beta_root(depth, alpha, beta)
			if (score <= alpha or score >= beta) and not self.time_up():
				if self.time_limit - (time.perf_counter() - self.start_time) > 0.3:
					score, bm = self.alpha_beta_root(depth, -1e9, 1e9)
			if bm is not None:
				last_best, last_score = bm, score
			depth += 1
		return last_best if last_best else self.fallback_move()

	def alpha_beta_root(self, depth, alpha, beta):
		color = self.me
		moves = legal_moves(self.prev, self.start_board, color)
		if self.move_count < 4 and None in moves:
			moves.remove(None)
			moves.append(None)
		if (self.me == BLACK and self.prev and
			all(all(c==EMPTY for c in r) for r in self.prev) and
			all(all(c==EMPTY for c in r) for r in self.start_board)):
			for c in [(0,0),(0,4),(4,0),(4,4)]:
				if c in moves:
					moves.remove(c)
					moves.insert(0, c)
					break
		pv_move = self.pv.get(self.key(self.prev, self.start_board, color))
		if pv_move in moves:
			moves.remove(pv_move)
			moves.insert(0, pv_move)

		def score_root(m):
			sc = ordering_score(self.start_board, m, color)
			mid = self.move_id(m)
			sc += 0.01 * self.history.get(mid, 0)
			return sc
		moves.sort(key=score_root, reverse=True)

		best_val, best = -1e9, None
		for m in moves:
			if self.time_up(): break
			nb, _ = simulate_play(self.start_board, m, color)
			if nb is None: continue
			val = self.alpha_beta(self.prev, self.start_board, nb, 3 - color, depth - 1, alpha, beta)
			if val > best_val:
				best_val, best = val, m
				if val > alpha:
					alpha = val
					self.pv[self.key(self.prev, self.start_board, color)] = m
			if alpha >= beta: break
		self.best_move = best
		return best_val, best

	def quiescence(self, prev_board, cur_board, color, alpha, beta, qdepth=6, seen=None):
		if qdepth <= 0 or self.time_up():
			return evaluate(cur_board, self.me, self.move_count)

		key_seen = (tuple(map(tuple, cur_board)), color)
		if seen is None: seen = set()
		else:
			if key_seen in seen:
				return evaluate(cur_board, self.me, self.move_count)
		seen.add(key_seen)

		stand_pat = evaluate(cur_board, self.me, self.move_count)
		if color == self.me:
			if stand_pat > alpha: alpha = stand_pat
			if alpha >= beta: return alpha
			caps = capture_moves(prev_board, cur_board, color)
			if not caps: return stand_pat
			caps.sort(key=lambda t: ordering_score(cur_board, t[0], color), reverse=True)
			for m, nb in caps:
				v = self.quiescence(cur_board, nb, 3 - color, alpha, beta, qdepth - 1, seen)
				if v > alpha: alpha = v
				if alpha >= beta: break
			return alpha
		else:
			if stand_pat < beta: beta = stand_pat
			if alpha >= beta: return beta
			caps = capture_moves(prev_board, cur_board, color)
			if not caps: return stand_pat
			caps.sort(key=lambda t: ordering_score(cur_board, t[0], color), reverse=True)
			for m, nb in caps:
				v = self.quiescence(cur_board, nb, 3 - color, alpha, beta, qdepth - 1, seen)
				if v < beta: beta = v
				if alpha >= beta: break
			return beta


	def alpha_beta(self, prev_board, cur_board, next_board, color, depth, alpha, beta):
		alpha0, beta0 = alpha, beta
		if self.time_up():
			return evaluate(next_board, self.me, self.move_count)
		self.nodes += 1
		k = self.key(prev_board, next_board, color)
		hit = self.TT.get(k)
		if hit:
			d, flag, val, mv = hit
			if d >= depth:
				if flag == self.FLAG_EXACT: return val
				elif flag == self.FLAG_LOWER and val > alpha: alpha = val
				elif flag == self.FLAG_UPPER and val < beta: beta = val
				if alpha >= beta: return val
			if mv: self.pv[k] = mv
		if depth == 0:
			return self.quiescence(prev_board, next_board, color, alpha, beta, qdepth=10)
		moves = legal_moves(cur_board, next_board, color)
		pv_move = self.pv.get(k)
		if pv_move in moves:
			moves.remove(pv_move)
			moves.insert(0, pv_move)
		killers = self.killers[depth]
		for km in killers:
			if km and km in moves:
				moves.remove(km)
				moves.insert(0, km)
		def score_child(m):
			sc = ordering_score(next_board, m, color)
			mid = self.move_id(m)
			sc += 0.01 * self.history.get(mid, 0)
			if killers[0] == m: sc += 0.2
			if killers[1] == m: sc += 0.1
			return sc
		moves.sort(key=score_child, reverse=True)

		best_move = None
		if color == self.me:
			val = -1e9
			for m in moves:
				nb, _ = simulate_play(next_board, m, color)
				if nb is None: continue
				v = self.alpha_beta(cur_board, next_board, nb, 3 - color, depth - 1, alpha, beta)
				if v > val:
					val, best_move = v, m
				if v > alpha:
					alpha = v
					self.pv[k] = m
				if alpha >= beta:
					mid = self.move_id(m)
					if killers[0] != m:
						killers[1], killers[0] = killers[0], m
					self.history[mid] = self.history.get(mid, 0) + depth*depth
					break
			flag = self.FLAG_EXACT
			if val >= beta0: flag = self.FLAG_LOWER
			elif val <= alpha0: flag = self.FLAG_UPPER
			self.TT[k] = (depth, flag, val, best_move)
			return val
		else:
			val = 1e9
			for m in moves:
				nb, _ = simulate_play(next_board, m, color)
				if nb is None: continue
				v = self.alpha_beta(cur_board, next_board, nb, 3 - color, depth - 1, alpha, beta)
				if v < val:
					val, best_move = v, m
				if v < beta:
					beta = v
					self.pv[k] = m
				if alpha >= beta:
					mid = self.move_id(m)
					if killers[0] != m:
						killers[1], killers[0] = killers[0], m
					self.history[mid] = self.history.get(mid, 0) + depth*depth
					break
			flag = self.FLAG_EXACT
			if val <= alpha0: flag = self.FLAG_UPPER
			elif val >= beta0: flag = self.FLAG_LOWER
			self.TT[k] = (depth, flag, val, best_move)
			return val

	def fallback_move(self):
		moves = [m for m in legal_moves(self.prev, self.start_board, self.me) if m]
		if not moves: return None
		moves.sort(key=lambda m: ordering_score(self.start_board, m, self.me), reverse=True)
		return moves[0] if moves else None

def main():
	me, prev, cur = read_input()
	searcher = Searcher(me, prev, cur)
	move = searcher.search()
	write_output(move)


def pick_move(prev_board: Optional[Board], cur_board: Board, me: int, time_limit: Optional[float] = None) -> Move:
	"""Public helper for invoking the alpha-beta engine."""
	return Searcher(me, prev_board, cur_board, time_limit=time_limit).search()


if __name__ == "__main__":
	main()
